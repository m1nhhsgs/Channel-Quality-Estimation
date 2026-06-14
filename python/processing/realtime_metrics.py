"""Realtime metric helpers shared by GUI and future live mode."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev


REQUIRED_RAW_COLUMNS = {
    "rx_time_ms",
    "sequence_number",
    "tx_timestamp_ms",
    "rssi",
    "payload_size",
}

LEGACY_RX_FIELDS = [
    "record",
    "session_id",
    "rx_ms_receiver",
    "src_mac",
    "dst_mac",
    "seq",
    "tx_ms_sender",
    "gap",
    "total_missing",
    "name",
    "value",
    "rssi_dbm",
    "noise_floor_dbm",
    "snr_db",
    "channel",
    "sig_mode",
    "mcs",
    "rate",
]

SIMPLE_RX_FIELDS = [
    "rx_time_ms",
    "sequence_number",
    "tx_timestamp_ms",
    "rssi",
    "payload_size",
]


@dataclass(frozen=True)
class PacketRecord:
    rx_time_ms: float
    sequence_number: int
    tx_timestamp_ms: float
    rssi: float
    payload_size: int
    scenario_id: str = "unknown"
    packet_rate: float = 0.0


@dataclass(frozen=True)
class MetricSnapshot:
    rssi_mean: float
    rssi_std: float
    rssi_min: float
    rssi_max: float
    rssi_slope: float
    packet_loss_percent: float
    throughput_bps: float
    inter_arrival_mean_ms: float
    jitter_ms: float
    entropy_rssi: float
    link_margin_bps: float
    packet_rate_hz: float
    payload_size: int
    predicted_loss_percent: float
    channel_state: str


def load_raw_csv(path: str | Path) -> list[PacketRecord]:
    """Load raw replay CSV with the XGBoost interface schema.

    Legacy receiver logs from the earlier firmware are also accepted so collected
    runs remain usable during the transition.
    """
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        legacy_columns = {"rx_ms_receiver", "seq", "tx_ms_sender", "rssi_dbm"}
        has_raw_schema = REQUIRED_RAW_COLUMNS.issubset(fieldnames)
        has_legacy_schema = legacy_columns.issubset(fieldnames)
        if not has_raw_schema and not has_legacy_schema:
            missing = REQUIRED_RAW_COLUMNS - fieldnames
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required columns: {missing_text}")

        records: list[PacketRecord] = []
        for row_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            try:
                if has_raw_schema:
                    records.append(
                        PacketRecord(
                            rx_time_ms=float(row["rx_time_ms"]),
                            sequence_number=int(float(row["sequence_number"])),
                            tx_timestamp_ms=float(row["tx_timestamp_ms"]),
                            rssi=float(row["rssi"]),
                            payload_size=int(float(row["payload_size"])),
                            scenario_id=(row.get("scenario_id") or "unknown").strip()
                            or "unknown",
                            packet_rate=float(row.get("packet_rate") or 0.0),
                        )
                    )
                else:
                    records.append(
                        PacketRecord(
                            rx_time_ms=float(row["rx_ms_receiver"]),
                            sequence_number=int(float(row["seq"])),
                            tx_timestamp_ms=float(row["tx_ms_sender"]),
                            rssi=float(row["rssi_dbm"]),
                            payload_size=int(float(row.get("payload_size") or 28)),
                            scenario_id=(row.get("session_id") or "unknown").strip()
                            or "unknown",
                            packet_rate=float(row.get("packet_rate") or 0.0),
                        )
                    )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value at CSV row {row_number}") from exc

    return sorted(records, key=lambda record: (record.rx_time_ms, record.sequence_number))


def packet_loss_percent(records: list[PacketRecord]) -> float:
    if not records:
        return 0.0
    sequences = [record.sequence_number for record in records]
    expected = max(sequences) - min(sequences) + 1
    if expected <= 0:
        return 0.0
    received = len(set(sequences))
    missing = max(expected - received, 0)
    return missing / expected * 100.0


def entropy(values: list[float], bins: int = 8) -> float:
    if len(values) <= 1:
        return 0.0
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return 0.0

    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = int((value - low) / width)
        counts[min(index, bins - 1)] += 1

    total = len(values)
    result = 0.0
    for count in counts:
        if count:
            probability = count / total
            result -= probability * math.log2(probability)
    return result


def entropy_by_width(values: list[float], bin_width: float = 1.0) -> float:
    if not values:
        return 0.0
    counts: dict[int, int] = {}
    for value in values:
        key = math.floor(value / bin_width)
        counts[key] = counts.get(key, 0) + 1
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def linear_slope_per_second(times_ms: list[float], values: list[float]) -> float:
    if len(times_ms) <= 1:
        return 0.0

    times_sec = [time_ms / 1000.0 for time_ms in times_ms]
    x_mean = mean(times_sec)
    y_mean = mean(values)
    denominator = sum((x - x_mean) ** 2 for x in times_sec)
    if math.isclose(denominator, 0.0):
        return 0.0
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(times_sec, values))
    return numerator / denominator


def parse_serial_packet(
    line: str,
    *,
    scenario_id: str = "live",
    default_payload_size: int = 28,
    default_packet_rate: float = 20.0,
) -> PacketRecord | None:
    parts = [part.strip() for part in line.split(",")]

    if len(parts) == len(SIMPLE_RX_FIELDS) and parts[0] != "rx_time_ms":
        try:
            return PacketRecord(
                rx_time_ms=float(parts[0]),
                sequence_number=int(float(parts[1])),
                tx_timestamp_ms=float(parts[2]),
                rssi=float(parts[3]),
                payload_size=int(float(parts[4])),
                scenario_id=scenario_id,
                packet_rate=default_packet_rate,
            )
        except ValueError:
            return None

    if len(parts) == len(LEGACY_RX_FIELDS) and parts[0] == "RX":
        row = dict(zip(LEGACY_RX_FIELDS, parts))
        try:
            return PacketRecord(
                rx_time_ms=float(row["rx_ms_receiver"]),
                sequence_number=int(float(row["seq"])),
                tx_timestamp_ms=float(row["tx_ms_sender"]),
                rssi=float(row["rssi_dbm"]),
                payload_size=default_payload_size,
                scenario_id=scenario_id or row.get("session_id", "live"),
                packet_rate=default_packet_rate,
            )
        except ValueError:
            return None

    return None


def infer_packet_rate(records: list[PacketRecord], fallback: float = 20.0) -> float:
    configured_rates = [record.packet_rate for record in records if record.packet_rate > 0]
    if configured_rates:
        return mean(configured_rates)

    sorted_records = sorted(records, key=lambda record: record.rx_time_ms)
    diffs = [
        (later.rx_time_ms - earlier.rx_time_ms) / 1000.0
        for earlier, later in zip(sorted_records, sorted_records[1:])
        if later.rx_time_ms > earlier.rx_time_ms
    ]
    if diffs:
        ordered = sorted(diffs)
        median_interval = ordered[len(ordered) // 2]
        if median_interval > 0:
            return 1.0 / median_interval
    return fallback


def model_feature_row(
    records: list[PacketRecord],
    *,
    window_size_sec: float = 2.0,
    default_packet_rate: float = 20.0,
    default_payload_size: int = 28,
) -> dict[str, float]:
    if not records:
        return {}

    sorted_records = sorted(records, key=lambda record: record.rx_time_ms)
    times_ms = [record.rx_time_ms for record in sorted_records]
    rssi_values = [record.rssi for record in sorted_records]
    payload_sizes = [
        record.payload_size if record.payload_size > 0 else default_payload_size
        for record in sorted_records
    ]
    inter_arrivals = [
        later.rx_time_ms - earlier.rx_time_ms
        for earlier, later in zip(sorted_records, sorted_records[1:])
        if later.rx_time_ms >= earlier.rx_time_ms
    ]

    rssi_min = min(rssi_values)
    rssi_max = max(rssi_values)
    return {
        "rssi_mean_2s": mean(rssi_values),
        "rssi_min_2s": rssi_min,
        "rssi_max_2s": rssi_max,
        "rssi_std_2s": pstdev(rssi_values) if len(rssi_values) > 1 else 0.0,
        "rssi_range_2s": rssi_max - rssi_min,
        "rssi_last": rssi_values[-1],
        "rssi_delta_2s": rssi_values[-1] - rssi_values[0],
        "rssi_slope_2s": linear_slope_per_second(times_ms, rssi_values),
        "packet_loss_past_2s": packet_loss_percent(sorted_records),
        "throughput_past_2s": sum(payload_sizes) * 8.0 / max(window_size_sec, 1e-9),
        "inter_arrival_mean_2s": mean(inter_arrivals) if inter_arrivals else 0.0,
        "jitter_2s": pstdev(inter_arrivals) if len(inter_arrivals) > 1 else 0.0,
        "entropy_rssi_2s": entropy_by_width(rssi_values, 1.0),
        "payload_size": mean(payload_sizes),
        "packet_rate": infer_packet_rate(sorted_records, default_packet_rate),
    }


def calculate_metrics(
    records: list[PacketRecord],
    *,
    noise_floor_dbm: float = -95.0,
    bandwidth_hz: float = 1_000_000.0,
    efficiency: float = 0.25,
) -> MetricSnapshot:
    if not records:
        return MetricSnapshot(
            rssi_mean=0.0,
            rssi_std=0.0,
            rssi_min=0.0,
            rssi_max=0.0,
            rssi_slope=0.0,
            packet_loss_percent=0.0,
            throughput_bps=0.0,
            inter_arrival_mean_ms=0.0,
            jitter_ms=0.0,
            entropy_rssi=0.0,
            link_margin_bps=0.0,
            packet_rate_hz=0.0,
            payload_size=0,
            predicted_loss_percent=0.0,
            channel_state="Good",
        )

    sorted_records = sorted(records, key=lambda record: record.rx_time_ms)
    times = [record.rx_time_ms for record in sorted_records]
    rssi_values = [record.rssi for record in sorted_records]
    payload_sizes = [record.payload_size for record in sorted_records]

    duration_sec = max((times[-1] - times[0]) / 1000.0, 0.001)
    inter_arrivals = [
        later.rx_time_ms - earlier.rx_time_ms
        for earlier, later in zip(sorted_records, sorted_records[1:])
        if later.rx_time_ms >= earlier.rx_time_ms
    ]

    loss = packet_loss_percent(sorted_records)
    throughput = sum(payload_sizes) * 8.0 / duration_sec
    packet_rate = len(sorted_records) / duration_sec
    rssi_avg = mean(rssi_values)
    snr_db = rssi_avg - noise_floor_dbm
    snr_linear = 10 ** (snr_db / 10.0)
    capacity_eff = efficiency * bandwidth_hz * math.log2(1.0 + snr_linear)
    offered_rate = packet_rate * mean(payload_sizes) * 8.0
    link_margin = capacity_eff - offered_rate
    slope = linear_slope_per_second(times, rssi_values)
    jitter = pstdev(inter_arrivals) if len(inter_arrivals) > 1 else 0.0
    inter_arrival_mean = mean(inter_arrivals) if inter_arrivals else 0.0
    predicted_loss = heuristic_prediction(
        loss=loss,
        rssi_mean=rssi_avg,
        rssi_slope=slope,
        jitter_ms=jitter,
        link_margin_bps=link_margin,
    )

    return MetricSnapshot(
        rssi_mean=rssi_avg,
        rssi_std=pstdev(rssi_values) if len(rssi_values) > 1 else 0.0,
        rssi_min=min(rssi_values),
        rssi_max=max(rssi_values),
        rssi_slope=slope,
        packet_loss_percent=loss,
        throughput_bps=throughput,
        inter_arrival_mean_ms=inter_arrival_mean,
        jitter_ms=jitter,
        entropy_rssi=entropy(rssi_values),
        link_margin_bps=link_margin,
        packet_rate_hz=packet_rate,
        payload_size=round(mean(payload_sizes)),
        predicted_loss_percent=predicted_loss,
        channel_state=channel_state(predicted_loss, link_margin),
    )


def heuristic_prediction(
    *,
    loss: float,
    rssi_mean: float,
    rssi_slope: float,
    jitter_ms: float,
    link_margin_bps: float,
) -> float:
    """Temporary predictor used until Lam exports the trained XGBoost package."""
    prediction = loss
    if rssi_mean < -80:
        prediction += 8.0
    elif rssi_mean < -70:
        prediction += 3.0

    if rssi_slope < -5:
        prediction += min(abs(rssi_slope) * 0.6, 10.0)
    if jitter_ms > 20:
        prediction += 5.0
    elif jitter_ms > 10:
        prediction += 2.0
    if link_margin_bps < 0:
        prediction += 10.0

    return max(0.0, min(prediction, 100.0))


def channel_state(predicted_loss_percent: float, link_margin_bps: float) -> str:
    if predicted_loss_percent >= 10.0 or link_margin_bps < 0:
        return "Critical"
    return "Good"


def state_from_loss(predicted_loss_percent: float) -> str:
    return "Critical" if predicted_loss_percent >= 10.0 else "Good"
