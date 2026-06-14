# Channel Quality Estimation

Du an nay xay dung he thong do, ghi log, trich xuat dac trung va du doan chat
luong kenh truyen ESP-NOW giua hai board ESP32. Mo hinh hien tai dung XGBoost
de du doan `packet_loss_future_2s`, sau do quy doi thanh 2 trang thai:

```text
Good:     packet_loss_future_2s < 10%
Critical: packet_loss_future_2s >= 10%
```

He thong gom 4 phan chinh:

```text
ESP32 TX/RX firmware  ->  Serial raw log
Serial logger         ->  data/raw/*.csv
Feature extraction    ->  data/features/ml_dataset.csv
XGBoost + GUI         ->  prediction, metrics, replay/live monitoring
```

## Ket qua hien tai

Model hien tai la XGBoost regressor khong dung metadata tu `data/sessions.csv`.
Model van du doan gia tri packet loss dang so, con trang thai `Good/Critical`
duoc suy ra bang nguong 10%.

Ket qua tren test set hien tai:

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

Bang ket qua de dua vao bao cao/slide nam tai:

```text
results/report_tables/report_tables.md
results/report_tables/report_tables.html
results/report_tables/*.csv
```

## Cau truc thu muc

```text
configs/              Cau hinh experiment/model
firmware/             Firmware ESP32 TX/RX va mo ta packet format
python/acquisition/   Script ghi log Serial tu RX
python/processing/    Trich xuat feature va tinh metric realtime
python/ml/            Train, predict, split dataset va runtime predictor
python/gui/           GUI replay/live
data/raw/             Raw CSV log tu RX
data/features/        Dataset feature cho ML
models/               XGBoost model va model_config
results/              Metrics, predictions, figures va report tables
docs/                 Tai lieu huong dan chay he thong
```

## Yeu cau moi truong

Can Python 3 va cac thu vien trong `requirements.txt`.

Cai dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Kiem tra nhanh:

```bash
python3 - <<'PY'
import pandas, joblib, xgboost, serial, PyQt5, pyqtgraph
print("dependencies ok")
PY
```

## Firmware ESP32

Firmware hien tai giu format cu dang dung trong du an.

Nap TX:

```text
firmware/tx_espnow/espnow_sender.ino
```

Nap RX:

```text
firmware/rx_espnow/espnow_receiver.ino
```

Can kiem tra:

```cpp
const uint8_t espNowChannel = 1;
const uint32_t sendIntervalMs = 50;
```

`sendIntervalMs = 50` tuong duong 20 packet/s. MAC peer trong TX phai la MAC
cua board RX.

RX in ra Serial theo format:

```csv
RX,session_id,rx_ms_receiver,src_mac,dst_mac,seq,tx_ms_sender,gap,total_missing,name,value,rssi_dbm,noise_floor_dbm,snr_db,channel,sig_mode,mcs,rate
```

Chi tiet format nam tai:

```text
firmware/packet_format.md
```

## Thu du lieu raw

Cam board RX vao may, tim Serial port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Vi du ghi mot session 6000 packet:

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

Output mac dinh:

```text
data/raw/run01.csv
data/sessions.csv
```

Neu gap loi permission voi `/dev/ttyUSB0`:

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

Khong mo Serial Monitor va logger cung luc, vi mot Serial port chi nen co mot
chuong trinh doc.

## Trich xuat dac trung

Tao dataset feature tu tat ca raw CSV:

```bash
python3 python/processing/feature_extraction.py 'data/raw/*.csv' \
  -o data/features/ml_dataset.csv \
  --include-labels \
  --packet-rate 20 \
  --payload-size 28
```

Thong so feature window hien tai:

```text
window_size = 2.0s
step_size   = 0.5s
horizon     = 2.0s
target      = packet_loss_future_2s
```

Feature chinh:

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

Trong do `entropy_rssi_2s` la dac trung lien quan den ly thuyet thong tin,
dung de mo ta muc do bien dong cua RSSI trong cua so 2 giay.

## Train model

Train lai XGBoost tu dataset feature:

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

Output:

```text
models/xgboost_packet_loss_pipeline.joblib
models/xgboost_model.json
models/model_config.json
results/metrics.json
results/test_predictions.csv
results/feature_importance.csv
results/figures/*.png
```

`models/model_config.json` dinh nghia feature order, target va nguong trang thai:

```text
Good:     packet_loss < 10%
Critical: packet_loss >= 10%
```

## Predict offline

Chay du doan tren feature CSV:

```bash
python3 python/ml/predict_xgboost_model.py data/features/ml_dataset.csv \
  -o results/predictions.csv \
  --model models/xgboost_packet_loss_pipeline.joblib \
  --config models/model_config.json
```

Output quan trong:

```text
predicted_packet_loss_future_2s
predicted_state
```

## GUI replay

Chay GUI voi CSV da thu:

```bash
python3 python/gui/gui_app.py --csv data/raw/run02.csv
```

GUI hien thi:

```text
RSSI
Packet loss hien tai
Packet loss du doan 2 giay toi
Throughput
Jitter
Entropy RSSI
Link margin
Trang thai Good/Critical
```

Neu model khong load duoc, GUI se dung heuristic fallback va hien thi loi tren
thanh trang thai model.

## GUI live

Cam RX da nap firmware, dam bao TX dang phat, roi chay:

```bash
python3 python/gui/gui_app.py \
  --live \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --packet-rate 20 \
  --payload-size 28 \
  --scenario-id live_demo
```

## Tao bang bao cao/slide

Bang ket qua hien tai da duoc tao trong:

```text
results/report_tables/
```

File nen dung:

```text
results/report_tables/report_tables.md
results/report_tables/report_tables.html
```

Cac CSV rieng:

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

Hinh ket qua nam trong:

```text
results/figures/
```

Quan trong nhat:

```text
results/figures/results.png
results/figures/confusion_matrix.png
results/figures/feature_importance.png
results/figures/prediction_vs_actual.png
results/figures/residual_histogram.png
```

## Lenh kiem tra nhanh

Kiem tra syntax Python:

```bash
python3 -m compileall python
```

Kiem tra feature extraction voi mot run:

```bash
python3 python/processing/feature_extraction.py data/raw/run02.csv \
  -o /tmp/run02_features.csv \
  --include-labels
```

Kiem tra predict:

```bash
python3 python/ml/predict_xgboost_model.py /tmp/run02_features.csv \
  -o /tmp/run02_predictions.csv \
  --model models/xgboost_packet_loss_pipeline.joblib \
  --config models/model_config.json
```

Kiem tra GUI offscreen:

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

## Ghi chu danh gia

Model hien tai du doan duoc xu huong packet loss nhung `R2` van am, nghia la
kha nang tong quat sang mot so run test con han che. Tuy vay, khi quy ve 2 lop
`Good/Critical`, recall cua lop `Critical` rat cao, phu hop voi bai toan can
phat hien som tinh trang kenh xau.

Neu muon cai thien tiep, nen thu them du lieu that o cac dieu kien trung gian,
dac biet cac run co packet loss gan nguong 10%.
