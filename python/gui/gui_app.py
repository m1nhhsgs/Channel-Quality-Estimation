"""Replay/live GUI for ESP-NOW channel quality prediction."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover - depends on local desktop env.
    print("Missing GUI dependency. Run: python3 -m pip install -r requirements.txt")
    raise SystemExit(1) from exc

from python.ml.predictor import XGBoostPacketLossPredictor
from python.processing.realtime_metrics import (
    MetricSnapshot,
    PacketRecord,
    calculate_metrics,
    load_raw_csv,
    model_feature_row,
    parse_serial_packet,
    state_from_loss,
)


DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "run02.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgboost_packet_loss_pipeline.joblib"
DEFAULT_CONFIG = PROJECT_ROOT / "models" / "model_config.json"


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title: str, unit: str = "", parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.unit = unit
        self.setObjectName("metricCard")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setObjectName("metricTitle")
        self.value_label = QtWidgets.QLabel("--")
        self.value_label.setObjectName("metricValue")
        self.value_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(f"{value}{self.unit}")


class ChannelMonitorWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        csv_path: Path | None,
        predictor: XGBoostPacketLossPredictor,
        live_port: str,
        baudrate: int,
        scenario_id: str,
        payload_size: int,
        packet_rate: float,
        auto_live: bool = False,
    ):
        super().__init__()
        self.setWindowTitle("ESP-NOW Channel Monitor")
        self.resize(1220, 780)

        self.predictor = predictor
        self.default_payload_size = payload_size
        self.default_packet_rate = packet_rate
        self.scenario_id = scenario_id
        self.serial_conn = None

        self.records: list[PacketRecord] = []
        self.visible_records: list[PacketRecord] = []
        self.replay_index = 0
        self.window_size_ms = 2000.0
        self.history_ms = 60000.0
        self.live_timeout_sec = 1.5
        self.stream_start_ms: float | None = None
        self.last_live_packet_wall_time: float | None = None
        self.live_signal_lost = False

        self.replay_timer = QtCore.QTimer(self)
        self.replay_timer.timeout.connect(self.advance_replay)
        self.live_timer = QtCore.QTimer(self)
        self.live_timer.timeout.connect(self.poll_live_serial)

        self.time_values: list[float] = []
        self.rssi_values: list[float] = []
        self.loss_values: list[float] = []
        self.predicted_loss_values: list[float] = []

        self._build_ui(live_port, baudrate)
        self._apply_style()
        self.update_model_label()

        if csv_path and csv_path.exists():
            self.load_csv(csv_path)
        elif DEFAULT_CSV.exists():
            self.load_csv(DEFAULT_CSV)

        if auto_live:
            QtCore.QTimer.singleShot(0, self.connect_live)

    def _build_ui(self, live_port: str, baudrate: int) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        self.setCentralWidget(central)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)

        self.open_button = QtWidgets.QPushButton("Open CSV")
        self.open_button.clicked.connect(self.open_csv_dialog)
        self.start_button = QtWidgets.QPushButton("Start Replay")
        self.start_button.clicked.connect(self.start_replay)
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_all)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_stream)

        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["50 ms", "100 ms", "200 ms", "500 ms"])
        self.speed_combo.setCurrentText("100 ms")
        self.speed_combo.currentTextChanged.connect(self.update_timer_interval)

        self.port_input = QtWidgets.QLineEdit(live_port)
        self.port_input.setPlaceholderText("/dev/ttyUSB0")
        self.port_input.setMinimumWidth(120)
        self.baud_input = QtWidgets.QLineEdit(str(baudrate))
        self.baud_input.setMaximumWidth(90)
        self.connect_button = QtWidgets.QPushButton("Connect Live")
        self.connect_button.clicked.connect(self.connect_live)
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_live)

        for widget in (
            self.open_button,
            self.start_button,
            self.pause_button,
            self.reset_button,
            self.speed_combo,
            self.port_input,
            self.baud_input,
            self.connect_button,
            self.disconnect_button,
        ):
            widget.setMinimumHeight(34)
            header.addWidget(widget)
        root.addLayout(header)

        info_bar = QtWidgets.QHBoxLayout()
        self.file_label = QtWidgets.QLabel("No CSV loaded")
        self.file_label.setObjectName("fileLabel")
        self.file_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.model_label = QtWidgets.QLabel("Model: --")
        self.model_label.setObjectName("fileLabel")
        info_bar.addWidget(self.file_label, 2)
        info_bar.addWidget(self.model_label, 1)
        root.addLayout(info_bar)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(12)
        root.addLayout(content, 1)

        charts = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(charts)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(10)

        pg.setConfigOptions(antialias=True)
        self.rssi_plot = pg.PlotWidget(title="RSSI")
        self.rssi_plot.setLabel("left", "dBm")
        self.rssi_plot.setLabel("bottom", "Time", "s")
        self.rssi_curve = self.rssi_plot.plot(pen=pg.mkPen("#2563eb", width=2))

        self.loss_plot = pg.PlotWidget(title="Packet Loss")
        self.loss_plot.setLabel("left", "Loss", "%")
        self.loss_plot.setLabel("bottom", "Time", "s")
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen("#dc2626", width=2))
        self.predicted_loss_curve = self.loss_plot.plot(
            pen=pg.mkPen("#f59e0b", width=2, style=QtCore.Qt.DashLine)
        )

        chart_layout.addWidget(self.rssi_plot, 1)
        chart_layout.addWidget(self.loss_plot, 1)
        content.addWidget(charts, 3)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        self.state_label = QtWidgets.QLabel("Good")
        self.state_label.setObjectName("stateLabel")
        self.state_label.setAlignment(QtCore.Qt.AlignCenter)
        self.state_label.setMinimumHeight(58)
        side_layout.addWidget(self.state_label)

        self.scenario_label = QtWidgets.QLabel("Scenario: --")
        self.scenario_label.setObjectName("sideText")
        self.progress_label = QtWidgets.QLabel("Rows: 0 / 0")
        self.progress_label.setObjectName("sideText")
        side_layout.addWidget(self.scenario_label)
        side_layout.addWidget(self.progress_label)

        metric_grid = QtWidgets.QGridLayout()
        metric_grid.setHorizontalSpacing(10)
        metric_grid.setVerticalSpacing(10)
        self.cards = {
            "rssi": MetricCard("RSSI Mean", " dBm"),
            "loss": MetricCard("Packet Loss", "%"),
            "pred": MetricCard("Pred Loss 2s", "%"),
            "throughput": MetricCard("Throughput", " bps"),
            "jitter": MetricCard("Jitter", " ms"),
            "entropy": MetricCard("Entropy", ""),
            "margin": MetricCard("Link Margin", " bps"),
            "rate": MetricCard("Packet Rate", " Hz"),
        }

        for index, card in enumerate(self.cards.values()):
            metric_grid.addWidget(card, index // 2, index % 2)
        side_layout.addLayout(metric_grid)
        side_layout.addStretch(1)
        content.addWidget(side, 2)

        self.statusBar().showMessage("Ready")

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f6f7f9;
                color: #111827;
            }
            QPushButton, QComboBox, QLineEdit {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 7px 11px;
                color: #111827;
            }
            QPushButton:hover {
                background: #eef2ff;
                border-color: #94a3b8;
            }
            QLabel#fileLabel, QLabel#sideText {
                color: #475569;
                font-size: 12px;
            }
            QFrame#metricCard {
                background: #ffffff;
                border: 1px solid #dbe2ea;
                border-radius: 8px;
            }
            QLabel#metricTitle {
                color: #64748b;
                font-size: 12px;
            }
            QLabel#metricValue {
                color: #0f172a;
                font-size: 20px;
                font-weight: 600;
            }
            QLabel#stateLabel {
                color: #ffffff;
                border-radius: 8px;
                font-size: 26px;
                font-weight: 700;
            }
            """
        )
        for plot in (self.rssi_plot, self.loss_plot):
            plot.setBackground("#ffffff")
            plot.showGrid(x=True, y=True, alpha=0.25)

    def update_model_label(self) -> None:
        if self.predictor.ready:
            self.model_label.setText(f"Model: {self.predictor.model_path.name}")
        else:
            error = self.predictor.error or "not loaded"
            self.model_label.setText(f"Model: heuristic fallback ({error})")

    def open_csv_dialog(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open replay CSV",
            str(PROJECT_ROOT / "data"),
            "CSV files (*.csv);;All files (*)",
        )
        if file_path:
            self.load_csv(Path(file_path))

    def load_csv(self, csv_path: Path) -> None:
        try:
            self.records = load_raw_csv(csv_path)
        except Exception as exc:  # noqa: BLE001 - show GUI-friendly error.
            QtWidgets.QMessageBox.critical(self, "Cannot load CSV", str(exc))
            return

        self.file_label.setText(str(csv_path))
        self.reset_stream()
        self.statusBar().showMessage(f"Loaded {len(self.records)} rows")

    def start_replay(self) -> None:
        if not self.records:
            self.statusBar().showMessage("Load a CSV first")
            return
        self.disconnect_live()
        self.replay_timer.start(self.current_interval_ms())
        self.statusBar().showMessage("Replay running")

    def pause_all(self) -> None:
        self.replay_timer.stop()
        self.live_timer.stop()
        self.statusBar().showMessage("Paused")

    def reset_stream(self) -> None:
        self.replay_timer.stop()
        self.replay_index = 0
        self.visible_records = []
        self.stream_start_ms = None
        self.live_signal_lost = False
        self.time_values = []
        self.rssi_values = []
        self.loss_values = []
        self.predicted_loss_values = []
        self.rssi_curve.setData([], [])
        self.loss_curve.setData([], [])
        self.predicted_loss_curve.setData([], [])
        self.rssi_plot.enableAutoRange(axis=pg.ViewBox.XAxis)
        self.loss_plot.enableAutoRange(axis=pg.ViewBox.XAxis)
        self.update_metric_cards(calculate_metrics([]))
        self.progress_label.setText(f"Rows: 0 / {len(self.records)}")
        self.scenario_label.setText("Scenario: --")
        self.statusBar().showMessage("Reset")

    def update_timer_interval(self) -> None:
        if self.replay_timer.isActive():
            self.replay_timer.start(self.current_interval_ms())

    def current_interval_ms(self) -> int:
        return int(self.speed_combo.currentText().split()[0])

    def connect_live(self) -> None:
        try:
            import serial

            port = self.port_input.text().strip()
            baudrate = int(self.baud_input.text().strip())
            if not port:
                self.statusBar().showMessage("Enter a serial port first")
                return
            self.disconnect_live()
            self.records = []
            self.reset_stream()
            self.serial_conn = serial.Serial(port, baudrate, timeout=0)
            self.last_live_packet_wall_time = time.monotonic()
            self.live_signal_lost = False
            self.live_timer.start(50)
            self.file_label.setText(f"Live: {port} @ {baudrate}")
            self.statusBar().showMessage("Live serial connected, waiting for packets")
        except Exception as exc:  # noqa: BLE001 - show GUI-friendly error.
            QtWidgets.QMessageBox.critical(self, "Cannot open serial port", str(exc))

    def disconnect_live(self) -> None:
        self.live_timer.stop()
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
        self.last_live_packet_wall_time = None
        self.live_signal_lost = False

    def poll_live_serial(self) -> None:
        if self.serial_conn is None:
            return
        received_packet = False
        for _ in range(100):
            try:
                raw = self.serial_conn.readline()
            except Exception as exc:  # noqa: BLE001 - serial disconnects surface here.
                self.handle_live_serial_error(exc)
                return
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            record = parse_serial_packet(
                line,
                scenario_id=self.scenario_id,
                default_payload_size=self.default_payload_size,
                default_packet_rate=self.default_packet_rate,
            )
            if record is not None:
                received_packet = True
                self.add_record(record)
        if received_packet:
            self.last_live_packet_wall_time = time.monotonic()
            if self.live_signal_lost:
                self.live_signal_lost = False
                self.statusBar().showMessage("Live serial receiving packets")
        else:
            self.check_live_timeout()

    def check_live_timeout(self) -> None:
        if self.serial_conn is None or self.last_live_packet_wall_time is None:
            return
        elapsed_sec = time.monotonic() - self.last_live_packet_wall_time
        if elapsed_sec >= self.live_timeout_sec:
            self.mark_live_disconnected(elapsed_sec)

    def handle_live_serial_error(self, exc: Exception) -> None:
        self.mark_live_disconnected(self.live_timeout_sec, error=str(exc))
        self.disconnect_live()

    def mark_live_disconnected(self, elapsed_sec: float, error: str | None = None) -> None:
        if self.live_signal_lost:
            return
        self.live_signal_lost = True
        self.set_channel_state("Disconnected")
        self.set_metric_cards_unavailable()
        if not self.records:
            self.progress_label.setText(f"Rows: {len(self.visible_records)} live (signal lost)")
        if error:
            self.statusBar().showMessage(f"Live serial disconnected: {error}")
        else:
            self.statusBar().showMessage(f"No live packets for {elapsed_sec:.1f}s")

    def advance_replay(self) -> None:
        if self.replay_index >= len(self.records):
            self.replay_timer.stop()
            self.statusBar().showMessage("Replay finished")
            return

        record = self.records[self.replay_index]
        self.replay_index += 1
        self.add_record(record)
        self.progress_label.setText(f"Rows: {self.replay_index} / {len(self.records)}")

    def add_record(self, record: PacketRecord) -> None:
        if self.serial_conn is not None:
            was_signal_lost = self.live_signal_lost
            self.last_live_packet_wall_time = time.monotonic()
            self.live_signal_lost = False
            if was_signal_lost:
                self.statusBar().showMessage("Live serial receiving packets")

        self.visible_records.append(record)
        self.visible_records = [
            item
            for item in self.visible_records
            if item.rx_time_ms >= record.rx_time_ms - self.history_ms
        ]

        window_records = [
            item
            for item in self.visible_records
            if item.rx_time_ms >= record.rx_time_ms - self.window_size_ms
        ]
        metrics = calculate_metrics(window_records)
        features = model_feature_row(
            window_records,
            window_size_sec=self.window_size_ms / 1000.0,
            default_packet_rate=self.default_packet_rate,
            default_payload_size=self.default_payload_size,
        )
        predicted = self.predictor.predict(features) if features else None
        if predicted is not None:
            metrics = replace(
                metrics,
                predicted_loss_percent=predicted,
                channel_state=state_from_loss(predicted),
            )

        self.append_chart_values(record, metrics)
        self.update_metric_cards(metrics)
        self.scenario_label.setText(f"Scenario: {record.scenario_id}")
        if not self.records:
            self.progress_label.setText(f"Rows: {len(self.visible_records)} live")

    def append_chart_values(self, record: PacketRecord, metrics: MetricSnapshot) -> None:
        if self.stream_start_ms is None:
            self.stream_start_ms = record.rx_time_ms

        time_sec = max(0.0, (record.rx_time_ms - self.stream_start_ms) / 1000.0)
        self.time_values.append(time_sec)
        self.rssi_values.append(record.rssi)
        self.loss_values.append(metrics.packet_loss_percent)
        self.predicted_loss_values.append(metrics.predicted_loss_percent)

        min_visible_time = max(0.0, time_sec - self.history_ms / 1000.0)
        points = [
            point
            for point in zip(
                self.time_values,
                self.rssi_values,
                self.loss_values,
                self.predicted_loss_values,
            )
            if point[0] >= min_visible_time
        ]

        max_points = 1200
        points = points[-max_points:]
        if points:
            (
                self.time_values,
                self.rssi_values,
                self.loss_values,
                self.predicted_loss_values,
            ) = [list(values) for values in zip(*points)]
        else:
            self.time_values = []
            self.rssi_values = []
            self.loss_values = []
            self.predicted_loss_values = []

        self.rssi_curve.setData(self.time_values, self.rssi_values)
        self.loss_curve.setData(self.time_values, self.loss_values)
        self.predicted_loss_curve.setData(self.time_values, self.predicted_loss_values)
        self.update_plot_x_range(time_sec)

    def update_plot_x_range(self, current_time_sec: float) -> None:
        history_sec = self.history_ms / 1000.0
        view_start = max(0.0, current_time_sec - history_sec)
        view_end = max(5.0, current_time_sec)
        if view_end - view_start < 5.0:
            view_end = view_start + 5.0
        self.rssi_plot.setXRange(view_start, view_end, padding=0)
        self.loss_plot.setXRange(view_start, view_end, padding=0)

    def update_metric_cards(self, metrics: MetricSnapshot) -> None:
        self.cards["rssi"].set_value(f"{metrics.rssi_mean:.1f}")
        self.cards["loss"].set_value(f"{metrics.packet_loss_percent:.1f}")
        self.cards["pred"].set_value(f"{metrics.predicted_loss_percent:.1f}")
        self.cards["throughput"].set_value(format_number(metrics.throughput_bps))
        self.cards["jitter"].set_value(f"{metrics.jitter_ms:.1f}")
        self.cards["entropy"].set_value(f"{metrics.entropy_rssi:.2f}")
        self.cards["margin"].set_value(format_number(metrics.link_margin_bps))
        self.cards["rate"].set_value(f"{metrics.packet_rate_hz:.1f}")
        self.set_channel_state(metrics.channel_state)

    def set_metric_cards_unavailable(self) -> None:
        for card in self.cards.values():
            card.set_value("--")

    def set_channel_state(self, state: str) -> None:
        colors = {
            "Good": "#15803d",
            "Critical": "#b91c1c",
            "Disconnected": "#475569",
        }
        self.state_label.setText(state)
        self.state_label.setStyleSheet(f"background: {colors.get(state, '#475569')};")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API.
        self.disconnect_live()
        super().closeEvent(event)


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ESP-NOW replay/live GUI")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Replay CSV path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="XGBoost pipeline .joblib path.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Model config JSON path.")
    parser.add_argument("--live", action="store_true", help="Connect live serial on startup.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Live serial port.")
    parser.add_argument("--baud", type=int, default=115200, help="Live serial baudrate.")
    parser.add_argument("--scenario-id", default="live", help="Scenario label for live packets.")
    parser.add_argument("--payload-size", type=int, default=28, help="Payload size for legacy firmware packets.")
    parser.add_argument("--packet-rate", type=float, default=20.0, help="Configured packet rate for live packets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictor = XGBoostPacketLossPredictor(args.model, args.config)
    predictor.load()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("ESP-NOW Channel Monitor")
    app.setWindowIcon(QtGui.QIcon())
    window = ChannelMonitorWindow(
        csv_path=args.csv,
        predictor=predictor,
        live_port=args.port,
        baudrate=args.baud,
        scenario_id=args.scenario_id,
        payload_size=args.payload_size,
        packet_rate=args.packet_rate,
        auto_live=args.live,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
