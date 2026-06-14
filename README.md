# Channel Quality Estimation

This project measures, logs, processes, and predicts ESP-NOW channel quality
between two ESP32 boards. The current machine learning model is an XGBoost
regressor that predicts `packet_loss_future_2s`. The numeric prediction is then
mapped to two channel states:

```text
Good:     packet_loss_future_2s < 10%
Critical: packet_loss_future_2s >= 10%
```

The system has four main stages:

```text
ESP32 TX/RX firmware  ->  Serial raw log
Serial logger         ->  data/raw/*.csv
Feature extraction    ->  data/features/ml_dataset.csv
XGBoost + GUI         ->  prediction, metrics, replay/live monitoring
```

## Current Results

The current model is an XGBoost regressor without session metadata from
`data/sessions.csv`. It predicts the packet loss percentage directly, while the
`Good/Critical` state is derived from the 10% threshold.

Current test-set results:

```text
Feature rows:          11511
Train rows:            8583
Test rows:             2928
MAE:                   8.3173
RMSE:                  13.1160
R2:                    -0.6171
Binary state accuracy: 0.7790
Binary F1 macro:       0.7605
Critical precision:    0.5366
Critical recall:       0.9813
```

Confusion matrix:

```text
                 Pred Good   Pred Critical
Actual Good         1548           633
Actual Critical       14           733
```

Report and slide tables are available at:

```text
results/report_tables/report_tables.md
results/report_tables/report_tables.html
results/report_tables/*.csv
```

## Project Structure

```text
configs/              Experiment and model configuration
firmware/             ESP32 TX/RX firmware and packet-format notes
python/acquisition/   Serial logging script
python/processing/    Feature extraction and realtime metric helpers
python/ml/            Training, prediction, dataset split, runtime predictor
python/gui/           Replay/live GUI application
data/raw/             Raw CSV logs from the receiver
data/features/        Machine learning feature dataset
models/               XGBoost model artifacts and model_config
results/              Metrics, predictions, figures, and report tables
docs/                 Additional system documentation
```

## Requirements

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Quick dependency check:

```bash
python3 - <<'PY'
import pandas, joblib, xgboost, serial, PyQt5, pyqtgraph
print("dependencies ok")
PY
```

## ESP32 Firmware

The current firmware keeps the legacy packet and serial-log format used by the
project.

Flash the TX board with:

```text
firmware/tx_espnow/espnow_sender.ino
```

Flash the RX board with:

```text
firmware/rx_espnow/espnow_receiver.ino
```

Check these firmware settings:

```cpp
const uint8_t espNowChannel = 1;
const uint32_t sendIntervalMs = 50;
```

`sendIntervalMs = 50` means 20 packets per second. The TX peer MAC address must
match the RX board MAC address.

The RX board prints serial lines in this format:

```csv
RX,session_id,rx_ms_receiver,src_mac,dst_mac,seq,tx_ms_sender,gap,total_missing,name,value,rssi_dbm,noise_floor_dbm,snr_db,channel,sig_mode,mcs,rate
```

Packet details are documented in:

```text
firmware/packet_format.md
```

## Raw Data Collection

Connect the RX board and find the serial port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Example command for logging one 6000-packet session:

```bash
python3 python/acquisition/serial_logger.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --session-id run01 \
  --location lab_room \
  --distance-m 1.0 \
  --link-type LOS \
  --obstacle none \
  --send-interval-ms 50 \
  --channel 1 \
  --expected-packets 6000
```

Default outputs:

```text
data/raw/run01.csv
data/sessions.csv
```

If `/dev/ttyUSB0` fails with a permission error:

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

Do not open Serial Monitor and the logger at the same time. A serial port should
only be read by one program at a time.

## Feature Extraction

Create the machine learning dataset from all raw CSV logs:

```bash
python3 python/processing/feature_extraction.py 'data/raw/*.csv' \
  -o data/features/ml_dataset.csv \
  --include-labels \
  --packet-rate 20 \
  --payload-size 28
```

Current sliding-window settings:

```text
window_size = 2.0s
step_size   = 0.5s
horizon     = 2.0s
target      = packet_loss_future_2s
```

Current model features:

```text
rssi_mean_2s
rssi_min_2s
rssi_max_2s
rssi_std_2s
rssi_range_2s
rssi_last
rssi_delta_2s
rssi_slope_2s
packet_loss_past_2s
throughput_past_2s
inter_arrival_mean_2s
jitter_2s
entropy_rssi_2s
payload_size
packet_rate
```

`entropy_rssi_2s` is the main information-theory-related feature. It describes
how much the RSSI values vary within a 2-second window.

## Model Training

Train the XGBoost model from the feature dataset:

```bash
python3 python/ml/train_xgboost_model.py data/features/ml_dataset.csv \
  --model-dir models \
  --results-dir results \
  --test-size 0.2 \
  --random-state 42 \
  --n-estimators 300 \
  --max-depth 5 \
  --learning-rate 0.05
```

Training outputs:

```text
models/xgboost_packet_loss_pipeline.joblib
models/xgboost_model.json
models/model_config.json
results/metrics.json
results/test_predictions.csv
results/feature_importance.csv
results/figures/*.png
```

`models/model_config.json` defines the feature order, prediction target, and
state thresholds:

```text
Good:     packet_loss < 10%
Critical: packet_loss >= 10%
```

## Offline Prediction

Run prediction on an extracted feature CSV:

```bash
python3 python/ml/predict_xgboost_model.py data/features/ml_dataset.csv \
  -o results/predictions.csv \
  --model models/xgboost_packet_loss_pipeline.joblib \
  --config models/model_config.json
```

Important output columns:

```text
predicted_packet_loss_future_2s
predicted_state
```

## GUI Replay Mode

Run the GUI with a collected CSV:

```bash
python3 python/gui/gui_app.py --csv data/raw/run02.csv
```

The GUI displays:

```text
RSSI
Current packet loss
Predicted packet loss for the next 2 seconds
Throughput
Jitter
RSSI entropy
Link margin
Good/Critical channel state
```

If the model cannot be loaded, the GUI uses a heuristic fallback and shows the
model error in the status area.

## GUI Live Mode

Connect the RX board, make sure the TX board is transmitting, then run:

```bash
python3 python/gui/gui_app.py \
  --live \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --packet-rate 20 \
  --payload-size 28 \
  --scenario-id live_demo
```

## Report and Slide Tables

Current report tables are available in:

```text
results/report_tables/
```

Recommended files:

```text
results/report_tables/report_tables.md
results/report_tables/report_tables.html
```

Individual CSV tables:

```text
01_dataset_overview.csv
02_model_metrics.csv
03_confusion_matrix.csv
04_state_thresholds.csv
05_full_dataset_state_distribution.csv
06_test_state_distribution.csv
07_test_performance_by_run.csv
08_feature_importance_top15.csv
09_binary_classification_details.csv
10_feature_dataset_by_run.csv
```

Figures are stored in:

```text
results/figures/
```

Most useful figures:

```text
results/figures/results.png
results/figures/confusion_matrix.png
results/figures/feature_importance.png
results/figures/prediction_vs_actual.png
results/figures/residual_histogram.png
```

## Quick Checks

Check Python syntax:

```bash
python3 -m compileall python
```

Check feature extraction on one run:

```bash
python3 python/processing/feature_extraction.py data/raw/run02.csv \
  -o /tmp/run02_features.csv \
  --include-labels
```

Check prediction:

```bash
python3 python/ml/predict_xgboost_model.py /tmp/run02_features.csv \
  -o /tmp/run02_predictions.csv \
  --model models/xgboost_packet_loss_pipeline.joblib \
  --config models/model_config.json
```

Check GUI startup in offscreen mode:

```bash
QT_QPA_PLATFORM=offscreen python3 - <<'PY'
import sys
from pathlib import Path
from PyQt5 import QtWidgets
from python.gui.gui_app import ChannelMonitorWindow
from python.ml.predictor import XGBoostPacketLossPredictor

predictor = XGBoostPacketLossPredictor(
    "models/xgboost_packet_loss_pipeline.joblib",
    "models/model_config.json",
)
assert predictor.load(), predictor.error

app = QtWidgets.QApplication(sys.argv)
window = ChannelMonitorWindow(
    csv_path=Path("data/raw/run02.csv"),
    predictor=predictor,
    live_port="/dev/ttyUSB0",
    baudrate=115200,
    scenario_id="test",
    payload_size=28,
    packet_rate=20.0,
)
for _ in range(5):
    window.advance_replay()
print("gui smoke ok", len(window.visible_records))
window.close()
app.quit()
PY
```

## Evaluation Notes

The current model captures packet-loss trends, but `R2` is still negative. This
means generalization to some held-out runs is limited. However, after converting
the output to the binary `Good/Critical` state, the `Critical` recall is high,
which is useful for detecting poor channel conditions early.

The most useful next improvement is collecting more real data near the 10%
packet-loss threshold.
