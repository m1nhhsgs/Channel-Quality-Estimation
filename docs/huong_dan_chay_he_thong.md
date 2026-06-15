# Huong dan chay he thong ESP-NOW + XGBoost

Tai lieu nay mo ta cach chay he thong hien tai trong repo: firmware cu dang dung
`DataPacket`, logger luu raw CSV format `RX,...`, model chinh la XGBoost du doan
`packet_loss_future_2s`.

## 1. Cai moi truong Python

Tu thu muc project:

```bash
python3 -m pip install -r requirements.txt
```

Kiem tra cac thu vien quan trong:

```bash
python3 - <<'PY'
import pandas, joblib, xgboost, PyQt5, pyqtgraph, serial
print("dependencies ok")
PY
```

## 2. Nap firmware

Nap TX:

```text
firmware/tx_espnow/espnow_sender.ino
```

Nap RX:

```text
firmware/rx_espnow/espnow_receiver.ino
```

Kiem tra trong sender:

```cpp
uint8_t peerMac[] = { ... };
```

Gia tri nay phai la MAC cua board RX. TX va RX phai cung:

```cpp
const uint8_t espNowChannel = 1;
```

Firmware hien tai gui 20 packet/s:

```cpp
const uint32_t sendIntervalMs = 50;
```

## 3. Thu du lieu live vao CSV

Kiem tra port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Vi du thu 6000 packet cho `run01`:

```bash
python3 python/acquisition/serial_logger.py --port /dev/ttyUSB0 --baud 115200 --session-id run01 --location lab_room --distance-m 1.0 --link-type LOS --obstacle none --send-interval-ms 50 --channel 1 --expected-packets 6000
```

Output mac dinh:

```text
data/raw/run01.csv
data/sessions.csv
```

Neu bi loi permission voi `/dev/ttyUSB0`:

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

## 4. Tao feature dataset

Tao dataset tu tat ca raw CSV:

```bash
python3 python/processing/feature_extraction.py 'data/raw/*.csv' -o data/features/ml_dataset.csv --include-labels
```

Output:

```text
data/features/ml_dataset.csv
```

Feature window hien tai:

```text
window_size = 2.0s
step_size   = 0.5s
horizon     = 2.0s
```

Target:

```text
packet_loss_future_2s
```

## 5. Chay predict offline

Dung model da train san:

```bash
python3 python/ml/predict_xgboost_model.py data/features/ml_dataset.csv -o results/predictions.csv --model models/xgboost_packet_loss_pipeline.joblib --config models/model_config.json
```

Output:

```text
results/predictions.csv
```

Cot quan trong:

```text
predicted_packet_loss_future_2s
predicted_state
```

## 6. Train lai model neu can

Train truc tiep tu dataset:

```bash
python3 python/ml/train_xgboost_model.py data/features/ml_dataset.csv --model-dir models --results-dir results
```

Hoac chia train/test theo scenario truoc:

```bash
python3 python/ml/split_dataset.py data/features/ml_dataset.csv --test-scenario-id run12 --train-output data/features/train_dataset.csv --test-output data/features/test_dataset.csv
python3 python/ml/train_xgboost_model.py --train-csv data/features/train_dataset.csv --test-csv data/features/test_dataset.csv --model-dir models --results-dir results
```

Model output:

```text
models/xgboost_packet_loss_pipeline.joblib
models/xgboost_model.json
models/model_config.json
```

## 7. Chay GUI Replay Mode

Chay voi CSV da thu:

```bash
python3 python/gui/gui_app.py --csv data/raw/run02.csv
```

GUI se:

- doc CSV replay
- tinh metric trong window 2 giay
- load `models/xgboost_packet_loss_pipeline.joblib`
- hien thi packet loss hien tai va packet loss du doan 2 giay toi
- hien thi trang thai Good / Critical

Neu model khong load duoc, GUI se hien thi fallback heuristic tren thanh `Model`.

## 8. Chay GUI Live Mode

Cam RX da nap firmware vao may, dam bao TX dang phat, roi chay:

```bash
python3 python/gui/gui_app.py --live --port /dev/ttyUSB0 --baud 115200 --packet-rate 20 --payload-size 28 --scenario-id live_demo
```

GUI se doc truc tiep Serial tu RX. Khong mo Serial Monitor hoac logger cung luc,
vi moi port Serial chi nen co mot chuong trinh dang doc.

Neu RX khong nhan duoc packet moi trong khoang 1.5 giay, GUI se hien thi
`Disconnected`. Khi packet quay lai, GUI tu cap nhat lai trang thai kenh ma
khong can bam reconnect.

## 9. Cac file chinh

```text
python/acquisition/serial_logger.py        Thu du lieu Serial
python/processing/feature_extraction.py    Tao feature CSV
python/ml/predict_xgboost_model.py         Predict offline
python/ml/train_xgboost_model.py           Train lai XGBoost
python/ml/predictor.py                     API du doan cho GUI
python/gui/gui_app.py                      GUI replay/live
models/xgboost_packet_loss_pipeline.joblib Model runtime cho GUI
models/model_config.json                   Feature order cua model
```

## 10. Lenh test nhanh

```bash
python3 -m compileall python
python3 python/processing/feature_extraction.py data/raw/run02.csv -o /tmp/run02_features.csv --include-labels
python3 python/ml/predict_xgboost_model.py /tmp/run02_features.csv -o /tmp/run02_predictions.csv --model models/xgboost_packet_loss_pipeline.joblib --config models/model_config.json
QT_QPA_PLATFORM=offscreen python3 - <<'PY'
import sys
from pathlib import Path
from PyQt5 import QtWidgets
from python.gui.gui_app import ChannelMonitorWindow
from python.ml.predictor import XGBoostPacketLossPredictor
p = XGBoostPacketLossPredictor("models/xgboost_packet_loss_pipeline.joblib", "models/model_config.json")
assert p.load(), p.error
app = QtWidgets.QApplication(sys.argv)
w = ChannelMonitorWindow(csv_path=Path("data/raw/run02.csv"), predictor=p, live_port="/dev/ttyUSB0", baudrate=115200, scenario_id="test", payload_size=28, packet_rate=20.0)
for _ in range(5):
    w.advance_replay()
print("gui smoke ok", len(w.visible_records))
w.close()
app.quit()
PY
```
