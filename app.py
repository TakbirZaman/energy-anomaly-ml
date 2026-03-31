"""
FastAPI Inference Server — LSTM Autoencoder Anomaly Detection
=============================================================
Loads the trained model + metadata produced by lstm_anomaly_detection.py
and exposes a REST API for real-time anomaly scoring of sensor windows.

When an anomaly is detected, the reading window is summarised and sent to
a configurable LLM (Anthropic Claude, OpenAI, or Google Gemini) which
returns a plain-English explanation for a factory technician.

Endpoints
---------
  POST /predict     — score a sensor window; includes LLM explanation on anomaly
  GET  /health      — liveness probe (model loaded, threshold set, LLM configured)
  GET  /model/info  — architecture & configuration details

Environment Variables
---------------------
  MODEL_CHECKPOINT   path to best_model.pt        (default: best_model.pt)
  MODEL_CONFIG       path to model_config.json     (default: model_config.json)
  HOST               bind host                     (default: 0.0.0.0)
  PORT               bind port                     (default: 8000)

  LLM_PROVIDER       "anthropic" | "openai" | "gemini"  (default: anthropic)
  LLM_API_KEY        API key for the chosen provider     (required for explanations)
  LLM_MODEL          override the default model per-provider (optional)
  LLM_TIMEOUT        seconds to wait for LLM response   (default: 15)

Quick Start
-----------
  # 1. Train and export artefacts
  python lstm_anomaly_detection.py

  # 2. Set your API key
  export LLM_PROVIDER=anthropic          # or openai / gemini
  export LLM_API_KEY=sk-ant-...

  # 3. Start the server
  uvicorn app:app --reload

  # 4. Test
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"readings": [[[5.0,5.0,-5.0,-5.0,5.0,-5.0], ...]]}'
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator, model_validator

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("anomaly_api")


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture  (mirrors lstm_anomaly_detection.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        seq_len: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_rep)
        return self.output_layer(out)


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        seq_len: int = 30,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.encoder = Encoder(n_features, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, n_features, seq_len, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ──────────────────────────────────────────────────────────────────────────────
# Application State
# ──────────────────────────────────────────────────────────────────────────────
class AppState:
    model: LSTMAutoencoder | None = None
    threshold: float | None = None
    device: torch.device = torch.device("cpu")
    config: dict[str, Any] = {}
    loaded_at: str = ""
    # LLM
    llm_provider: str = "anthropic"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout: float = 15.0


state = AppState()

DEFAULT_CONFIG: dict[str, Any] = {
    "n_features":  6,
    "hidden_size": 64,
    "seq_len":     30,
    "num_layers":  2,
    "dropout":     0.2,
    "threshold":   None,
    "scaler_mean": None,
    "scaler_std":  None,
}

# Default model names per provider
_PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-1.5-flash",
}


# ──────────────────────────────────────────────────────────────────────────────
# LLM Interpretation Helper
# ──────────────────────────────────────────────────────────────────────────────
def _build_sensor_summary(readings: list) -> str:
    """
    Compress the raw [1, T, F] window into a compact per-feature statistics
    table so the prompt stays well under 200 tokens regardless of window size.

    Example output:
      Sensor  1 — min: -1.230  max: +2.450  mean: +0.610  last: +1.980
      Sensor  2 — min: +0.100  max: +0.950  mean: +0.520  last: +0.880
    """
    window = np.array(readings[0], dtype=np.float32)   # (T, F)
    lines: list[str] = []
    for f in range(window.shape[1]):
        col = window[:, f]
        lines.append(
            f"  Sensor {f + 1:>2} — "
            f"min: {col.min():+.3f}  "
            f"max: {col.max():+.3f}  "
            f"mean: {col.mean():+.3f}  "
            f"last: {col[-1]:+.3f}"
        )
    return "\n".join(lines)


def _build_prompt(sensor_summary: str, reconstruction_error: float, threshold: float) -> str:
    """Assemble the technician-facing prompt sent to the LLM."""
    return (
        "The following industrial sensor readings have been flagged as an anomaly:\n\n"
        f"{sensor_summary}\n\n"
        f"Reconstruction error: {reconstruction_error:.4f}  (normal threshold: {threshold:.4f})\n\n"
        "Explain what might be happening to a factory technician in 2 sentences. "
        "Be specific about which sensors look abnormal and suggest a likely cause."
    )


async def _call_anthropic(prompt: str, client: httpx.AsyncClient) -> str:
    """Call the Anthropic Messages API (claude-sonnet-4-5 by default)."""
    response = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         state.llm_api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        json={
            "model":      state.llm_model,
            "max_tokens": 150,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=state.llm_timeout,
    )
    response.raise_for_status()
    return response.json()["content"][0]["text"].strip()


async def _call_openai(prompt: str, client: httpx.AsyncClient) -> str:
    """Call the OpenAI Chat Completions API (gpt-4o-mini by default)."""
    response = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {state.llm_api_key}",
            "Content-Type":  "application/json",
        },
        json={
            "model":      state.llm_model,
            "max_tokens": 150,
            "messages":   [{"role": "user", "content": prompt}],
        },
        timeout=state.llm_timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


async def _call_gemini(prompt: str, client: httpx.AsyncClient) -> str:
    """Call the Google Gemini generateContent API (gemini-1.5-flash by default)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{state.llm_model}:generateContent?key={state.llm_api_key}"
    )
    response = await client.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=state.llm_timeout,
    )
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


async def interpret_anomaly(
    readings: list,
    reconstruction_error: float,
    threshold: float,
) -> str | None:
    """
    Send the anomalous sensor window to the configured LLM and return a
    plain-English explanation for a factory technician.

    Design contract
    ---------------
    - Returns None (never raises) if no API key is set OR if the call fails.
    - Failures are logged at WARNING level so the /predict response is always
      returned — an unavailable LLM is never a hard error.
    - Uses httpx.AsyncClient so the event loop is never blocked.
    - Prompt is compressed to a per-sensor statistics table (min/max/mean/last)
      so it remains token-efficient regardless of window length.
    """
    if not state.llm_api_key:
        logger.warning(
            "LLM explanation skipped — LLM_API_KEY not set. "
            "Export LLM_API_KEY=<your-key> to enable explanations."
        )
        return None

    sensor_summary = _build_sensor_summary(readings)
    prompt = _build_prompt(sensor_summary, reconstruction_error, threshold)

    logger.debug("LLM prompt:\n%s", prompt)

    try:
        async with httpx.AsyncClient() as client:
            provider = state.llm_provider
            if provider == "anthropic":
                explanation = await _call_anthropic(prompt, client)
            elif provider == "openai":
                explanation = await _call_openai(prompt, client)
            elif provider == "gemini":
                explanation = await _call_gemini(prompt, client)
            else:
                logger.warning("Unknown LLM_PROVIDER '%s' — skipping explanation.", provider)
                return None

        logger.info("LLM explanation received (%d chars)", len(explanation))
        return explanation

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "LLM API HTTP %d — explanation unavailable. Body: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
    except httpx.TimeoutException:
        logger.warning(
            "LLM API timed out after %.1fs — explanation unavailable.",
            state.llm_timeout,
        )
    except Exception as exc:
        logger.warning(
            "LLM call failed (%s: %s) — explanation unavailable.",
            type(exc).__name__, exc,
        )

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Startup / Shutdown lifespan
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; configure LLM; release on shutdown."""
    checkpoint_path = Path(os.getenv("MODEL_CHECKPOINT", "best_model.pt"))
    config_path     = Path(os.getenv("MODEL_CONFIG",     "model_config.json"))

    # ── 1. Model config ──────────────────────────────────────────────────
    cfg = {**DEFAULT_CONFIG}
    if config_path.exists():
        with open(config_path) as f:
            cfg.update(json.load(f))
        logger.info("Loaded model config from %s", config_path)
    else:
        logger.warning(
            "Config '%s' not found — using defaults. Run lstm_anomaly_detection.py first.",
            config_path,
        )
    state.config = cfg

    # ── 2. Device ────────────────────────────────────────────────────────
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Inference device: %s", state.device)

    # ── 3. Model weights ─────────────────────────────────────────────────
    if not checkpoint_path.exists():
        logger.error("Checkpoint '%s' not found — model will not be loaded.", checkpoint_path)
    else:
        try:
            model = LSTMAutoencoder(
                n_features=cfg["n_features"],
                hidden_size=cfg["hidden_size"],
                seq_len=cfg["seq_len"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=state.device))
            model.to(state.device)
            model.eval()
            state.model = model
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                "Model loaded — %s | params: %s | seq_len: %d | features: %d",
                checkpoint_path, f"{total_params:,}", cfg["seq_len"], cfg["n_features"],
            )
        except Exception as exc:
            logger.exception("Failed to load model: %s", exc)

    # ── 4. Anomaly threshold ─────────────────────────────────────────────
    if cfg.get("threshold") is not None:
        state.threshold = float(cfg["threshold"])
        logger.info("Anomaly threshold: %.6f", state.threshold)
    else:
        logger.warning("No threshold in config — set 'threshold' in model_config.json.")

    # ── 5. LLM configuration ─────────────────────────────────────────────
    state.llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    state.llm_api_key  = os.getenv("LLM_API_KEY") or None
    state.llm_timeout  = float(os.getenv("LLM_TIMEOUT", "15"))
    state.llm_model    = (
        os.getenv("LLM_MODEL")
        or _PROVIDER_DEFAULTS.get(state.llm_provider, "claude-sonnet-4-5")
    )

    if state.llm_api_key:
        logger.info(
            "LLM configured — provider: %s | model: %s | timeout: %.0fs",
            state.llm_provider, state.llm_model, state.llm_timeout,
        )
    else:
        logger.warning(
            "LLM_API_KEY not set — anomaly explanations disabled. "
            "Export LLM_API_KEY to enable them."
        )

    state.loaded_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    yield

    logger.info("Server shutting down — releasing model.")
    state.model = None


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LSTM Autoencoder — Anomaly Detection API",
    description=(
        "Real-time anomaly scoring for multivariate sensor time-series.\n\n"
        "When an anomaly is detected the endpoint automatically queries an LLM "
        "(Anthropic / OpenAI / Gemini) for a plain-English explanation targeted "
        "at a factory technician.\n\n"
        "Set **`LLM_PROVIDER`** and **`LLM_API_KEY`** environment variables to "
        "enable explanations."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Sensor window for anomaly scoring.
    readings : 3-D list shaped  [1, sequence_length, n_features]
    """

    readings: list[list[list[float]]]

    @field_validator("readings")
    @classmethod
    def validate_outer_batch(cls, v: list) -> list:
        if not v:
            raise ValueError("'readings' must not be empty.")
        if len(v) != 1:
            raise ValueError(
                f"Batch size must be 1 (got {len(v)}). Send one window per request."
            )
        return v

    @model_validator(mode="after")
    def validate_inner_shape(self) -> "PredictRequest":
        seq = self.readings[0]
        if not seq:
            raise ValueError("The window (readings[0]) must not be empty.")

        n_feats = len(seq[0])
        for t, step in enumerate(seq):
            if len(step) != n_feats:
                raise ValueError(
                    f"Inconsistent feature count at time-step {t}: "
                    f"expected {n_feats}, got {len(step)}."
                )

        if state.model is not None:
            if len(seq) != state.model.seq_len:
                raise ValueError(
                    f"Wrong sequence length: model expects {state.model.seq_len} "
                    f"time-steps, got {len(seq)}."
                )
            if n_feats != state.model.n_features:
                raise ValueError(
                    f"Wrong feature count: model expects {state.model.n_features} "
                    f"features, got {n_feats}."
                )
        return self


class PredictResponse(BaseModel):
    anomaly: bool
    reconstruction_error: float
    status: str                    # "Critical" | "Normal"
    threshold: float
    sequence_length: int
    n_features: int
    llm_explanation: str | None    # None when anomaly=False or LLM unavailable
    llm_provider: str | None       # which provider answered (None if not called)
    llm_model: str | None          # which model answered  (None if not called)
    latency_ms: float
    llm_latency_ms: float | None   # isolated LLM round-trip time


class HealthResponse(BaseModel):
    healthy: bool
    model_loaded: bool
    threshold_set: bool
    llm_configured: bool
    llm_provider: str
    llm_model: str
    device: str
    loaded_at: str
    message: str


class ModelInfoResponse(BaseModel):
    n_features: int
    hidden_size: int
    seq_len: int
    num_layers: int
    dropout: float
    threshold: float | None
    total_parameters: int
    device: str
    llm_provider: str
    llm_model: str
    llm_configured: bool


# ──────────────────────────────────────────────────────────────────────────────
# Global exception handler
# ──────────────────────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check server logs."},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, summary="Liveness probe", tags=["System"])
async def health() -> HealthResponse:
    """
    Returns server readiness.  `healthy` is True only when the model and
    threshold are both loaded.  LLM configuration is reported but does not
    affect `healthy` — predictions work without an LLM key.
    """
    model_loaded  = state.model     is not None
    threshold_set = state.threshold is not None
    healthy       = model_loaded and threshold_set

    return HealthResponse(
        healthy=healthy,
        model_loaded=model_loaded,
        threshold_set=threshold_set,
        llm_configured=bool(state.llm_api_key),
        llm_provider=state.llm_provider,
        llm_model=state.llm_model or "",
        device=str(state.device),
        loaded_at=state.loaded_at,
        message="OK" if healthy else "Model or threshold not loaded — check logs.",
    )


@app.get("/model/info", response_model=ModelInfoResponse, summary="Model & LLM details", tags=["System"])
async def model_info() -> ModelInfoResponse:
    """Returns architecture hyperparameters, parameter count, and LLM configuration."""
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Ensure best_model.pt exists.",
        )
    total_params = sum(p.numel() for p in state.model.parameters())
    cfg = state.config
    return ModelInfoResponse(
        n_features=cfg["n_features"],
        hidden_size=cfg["hidden_size"],
        seq_len=cfg["seq_len"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        threshold=state.threshold,
        total_parameters=total_params,
        device=str(state.device),
        llm_provider=state.llm_provider,
        llm_model=state.llm_model or "",
        llm_configured=bool(state.llm_api_key),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score a sensor window; get LLM explanation on anomaly",
    tags=["Inference"],
    responses={
        200: {"description": "Anomaly verdict + optional LLM explanation"},
        422: {"description": "Input shape / type validation error"},
        503: {"description": "Model or threshold not loaded"},
    },
)
async def predict(payload: PredictRequest) -> PredictResponse:
    """
    Accepts one sensor window and returns an anomaly verdict.

    **Input shape**: `[1, sequence_length, n_features]`

    **LLM explanation** (`llm_explanation` field)
    - Only triggered when `anomaly == True`
    - Returns `null` when `LLM_API_KEY` is not set or when the LLM call fails
    - A failed LLM call never causes the endpoint to error — the anomaly verdict
      is always delivered

    **Supported providers** (set via `LLM_PROVIDER` env var):
    - `anthropic` → Claude (default: claude-sonnet-4-5)
    - `openai`    → GPT    (default: gpt-4o-mini)
    - `gemini`    → Gemini (default: gemini-1.5-flash)
    """
    t_start = time.perf_counter()

    # ── Guards ───────────────────────────────────────────────────────────
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Ensure 'best_model.pt' exists (check /health).",
        )
    if state.threshold is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Threshold not configured. Add 'threshold' to model_config.json.",
        )

    # ── Build tensor ─────────────────────────────────────────────────────
    try:
        arr    = np.array(payload.readings, dtype=np.float32)
        tensor = torch.from_numpy(arr).to(state.device)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not convert readings to tensor: {exc}",
        )

    # ── Optional z-score normalisation ───────────────────────────────────
    scaler_mean = state.config.get("scaler_mean")
    scaler_std  = state.config.get("scaler_std")
    if scaler_mean is not None and scaler_std is not None:
        mean_t = torch.tensor(scaler_mean, dtype=torch.float32, device=state.device)
        std_t  = torch.tensor(scaler_std,  dtype=torch.float32, device=state.device)
        tensor = (tensor - mean_t) / (std_t + 1e-8)

    # ── Autoencoder inference ─────────────────────────────────────────────
    with torch.no_grad():
        reconstructed = state.model(tensor)

    # ── Reconstruction MSE ────────────────────────────────────────────────
    error_per_element    = nn.MSELoss(reduction="none")(reconstructed, tensor)
    reconstruction_error = float(error_per_element.mean().item())

    # ── Anomaly decision ──────────────────────────────────────────────────
    is_anomaly   = reconstruction_error > state.threshold
    alert_status = "Critical" if is_anomaly else "Normal"

    inference_ms = (time.perf_counter() - t_start) * 1000.0
    logger.info(
        "predict | error=%.6f | threshold=%.6f | anomaly=%s | status=%s | %.2fms",
        reconstruction_error, state.threshold, is_anomaly, alert_status, inference_ms,
    )

    # ── LLM explanation (anomalies only) ─────────────────────────────────
    llm_explanation:   str | None   = None
    llm_latency_ms:    float | None = None
    llm_provider_used: str | None   = None
    llm_model_used:    str | None   = None

    if is_anomaly:
        t_llm = time.perf_counter()
        llm_explanation = await interpret_anomaly(
            readings=payload.readings,
            reconstruction_error=reconstruction_error,
            threshold=state.threshold,
        )
        llm_latency_ms = round((time.perf_counter() - t_llm) * 1000.0, 1)

        if llm_explanation is not None:
            llm_provider_used = state.llm_provider
            llm_model_used    = state.llm_model

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0

    return PredictResponse(
        anomaly=is_anomaly,
        reconstruction_error=round(reconstruction_error, 8),
        status=alert_status,
        threshold=round(state.threshold, 8),
        sequence_length=tensor.shape[1],
        n_features=tensor.shape[2],
        llm_explanation=llm_explanation,
        llm_provider=llm_provider_used,
        llm_model=llm_model_used,
        latency_ms=round(total_latency_ms, 2),
        llm_latency_ms=llm_latency_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dev server entrypoint
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        log_level="info",
    )
