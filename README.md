# Energy Anomaly ML

Detects weird power usage with an LSTM model. Trained on time-series energy data.

What it does:
- Learns normal usage pattern
- Flags anomalies in new data
- Simple Flask/FastAPI demo with `app.py`

Stack: Python, PyTorch (LSTM), scikit-learn

Run it:
```bash
pip install -r requirements.txt
python app.py
# or
python lstm_anomaly_detection.py
```
