#!/usr/bin/env python3
"""
Extract sliding-window features for ESP-NOW channel-quality prediction.

The script intentionally uses only the Python standard library so it can run
even when pandas/numpy are not installed correctly.

Example:
    python extract_xgboost_features.py receiver_log.csv -o xgboost_features.csv
    python extract_xgboost_features.py data/raw/*.csv -o ml_dataset.csv --include-labels
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


TIME_COLUMNS = ("rx_time_ms", "rx_ms_receiver", "rx_timestamp_ms", "timestamp_ms", "time_ms")
ISO_TIME_COLUMNS = ("pc_timestamp", "timestamp", "time")
SESSION_COLUMNS = ("session_id", "scenario_id", "run_id")
SEQ_COLUMNS = ("sequence_number", "seq")
RSSI_COLUMNS = ("rssi", "rssi_dbm")
TX_TIME_COLUMNS = ("tx_timestamp_ms", "tx_ms_sender")
PAYLOAD_COLUMNS = ("payload_size", "payload_len")
PACKET_RATE_COLUMNS = ("packet_rate",)


# Main feature set from the XGBoost document, section 7.
FEATURE_COLUMNS = [
    "scenario_id",
    "window_start",
    "window_end",
    "rssi_mean_2s",
    "rssi_min_2s",
    "rssi_max_2s",
    "rssi_std_2s",
    "rssi_range_2s",
    "rssi_last",
    "rssi_delta_2s",
    "rssi_slope_2s",
    "packet_loss_past_2s",
    "throughput_past_2s",
    "inter_arrival_mean_2s",
    "jitter_2s",
    "entropy_rssi_2s",
    "payload_size",
    "packet_rate",
    "link_margin",
]

LABEL_COLUMNS = [
    "packet_loss_future_2s",
]


def first_existing(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    names = set(fieldnames)
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_time_seconds(row: dict[str, str], time_col: str | None, iso_time_col: str | None) -> float | None:
    if time_col:
        value = to_float(row.get(time_col))
        if value is not None:
            return value / 1000.0

    if iso_time_col and row.get(iso_time_col):
        text = row[iso_time_col].strip()
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None

    return None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def mode(values: list[float]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    max_count = max(counts.values())
    return min(value for value, count in counts.items() if count == max_count)


def slope(times: list[float], values: list[float]) -> float | None:
    if len(times) < 2 or len(values) < 2:
        return 0.0
    t0 = times[0]
    xs = [t - t0 for t in times]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom


def entropy(values: list[float], bin_width: float) -> float | None:
    if not values:
        return None
    bins = Counter(math.floor(value / bin_width) for value in values)
    total = len(values)
    return -sum((count / total) * math.log2(count / total) for count in bins.values())


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def add_stats(prefix: str, values: list[float], out: dict[str, Any]) -> None:
    out[f"{prefix}_mean_2s"] = mean(values)
    out[f"{prefix}_min_2s"] = min(values) if values else None
    out[f"{prefix}_max_2s"] = max(values) if values else None
    out[f"{prefix}_std_2s"] = std(values) if values else None


def packet_loss(rows: list[dict[str, Any]]) -> tuple[float, float, float, float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    gaps = [row["gap"] for row in rows if row.get("gap") is not None]
    if gaps:
        missing = sum(max(0.0, gap) for gap in gaps)
    else:
        seqs = [row["seq"] for row in rows if row.get("seq") is not None]
        if len(seqs) >= 2:
            missing = max(0.0, max(seqs) - min(seqs) + 1 - len(set(seqs)))
        else:
            missing = 0.0

    received = float(len(rows))
    expected = received + missing
    loss_rate = (missing / expected * 100.0) if expected > 0 else 0.0
    max_gap = max(gaps) if gaps else missing
    gap_mean = statistics.fmean(gaps) if gaps else 0.0
    gap_std = std(gaps) if gaps else 0.0
    return missing, expected, loss_rate, max_gap, gap_mean, gap_std


def infer_packet_rate(rows: list[dict[str, Any]], fallback: float) -> float:
    rates = [row["packet_rate"] for row in rows if row.get("packet_rate") is not None]
    if rates:
        return statistics.fmean(rates)

    times = [row["time_s"] for row in rows]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    if diffs:
        median_interval = statistics.median(diffs)
        if median_interval > 0:
            return 1.0 / median_interval

    return fallback


def link_features(
    snr_mean_db: float | None,
    throughput: float,
    bandwidth_hz: float | None,
    efficiency: float,
) -> tuple[float | None, float | None, float | None]:
    if snr_mean_db is None or bandwidth_hz is None or bandwidth_hz <= 0:
        return None, None, None
    snr_linear = 10 ** (snr_mean_db / 10.0)
    capacity = efficiency * bandwidth_hz * math.log2(1.0 + snr_linear)
    margin = capacity - throughput
    utilization = throughput / capacity if capacity > 0 else None
    return capacity, margin, utilization


def build_feature_row(
    source_file: str,
    session_id: str,
    window_rows: list[dict[str, Any]],
    window_start: float,
    window_end: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rssi = [row["rssi"] for row in window_rows if row.get("rssi") is not None]
    noise = [row["noise_floor"] for row in window_rows if row.get("noise_floor") is not None]
    snr = [row["snr"] for row in window_rows if row.get("snr") is not None]
    times = [row["time_s"] for row in window_rows]
    payloads = [
        row["payload_size"] if row.get("payload_size") is not None else args.payload_size
        for row in window_rows
    ]
    payloads = [payload for payload in payloads if payload is not None]

    out: dict[str, Any] = {
        "source_file": source_file,
        "session_id": session_id,
        "scenario_id": session_id,
        "window_start": window_start,
        "window_end": window_end,
        "received_count": len(window_rows),
    }

    if rssi:
        out["rssi_mean_2s"] = mean(rssi)
        out["rssi_min_2s"] = min(rssi)
        out["rssi_max_2s"] = max(rssi)
        out["rssi_std_2s"] = std(rssi)
        out["rssi_range_2s"] = max(rssi) - min(rssi)
        out["rssi_last"] = rssi[-1]
        out["rssi_delta_2s"] = rssi[-1] - rssi[0]
        out["rssi_slope_2s"] = slope(times, rssi)
        out["entropy_rssi_2s"] = entropy(rssi, args.rssi_bin_width)

    add_stats("noise_floor", noise, out)
    add_stats("snr", snr, out)
    out["snr_p10_2s"] = percentile(snr, 0.10)
    out["snr_p50_2s"] = percentile(snr, 0.50)
    out["snr_p90_2s"] = percentile(snr, 0.90)

    missing, expected, loss_rate, max_gap, gap_mean, gap_std = packet_loss(window_rows)
    out["missing_count_2s"] = missing
    out["expected_count_2s"] = expected
    out["packet_loss_past_2s"] = loss_rate
    out["max_gap_2s"] = max_gap
    out["gap_mean_2s"] = gap_mean
    out["gap_std_2s"] = gap_std

    duration = max(args.window_size, 1e-9)
    throughput = sum(payloads) * 8.0 / duration if payloads else 0.0
    out["throughput_past_2s"] = throughput

    inter_arrivals_ms = [
        (b - a) * 1000.0 for a, b in zip(times, times[1:]) if b >= a
    ]
    out["inter_arrival_mean_2s"] = mean(inter_arrivals_ms)
    out["jitter_2s"] = std(inter_arrivals_ms) if inter_arrivals_ms else 0.0

    out["packet_rate"] = infer_packet_rate(window_rows, args.packet_rate)
    out["payload_size"] = mean(payloads) if payloads else args.payload_size

    for col in ("channel", "mcs", "rate"):
        values = [row[col] for row in window_rows if row.get(col) is not None]
        if col == "channel":
            out[col] = mode(values)
        else:
            out[f"{col}_mean"] = mean(values)
            out[f"{col}_mode"] = mode(values)
            if col == "rate":
                out["rate_min"] = min(values) if values else None
                out["rate_max"] = max(values) if values else None

    sig_modes = [row["sig_mode"] for row in window_rows if row.get("sig_mode") is not None]
    out["sig_mode_mode"] = mode(sig_modes)

    capacity, margin, utilization = link_features(
        out.get("snr_mean_2s"),
        throughput,
        args.bandwidth_hz,
        args.capacity_efficiency,
    )
    out["link_capacity_eff"] = capacity
    out["link_margin"] = margin
    out["link_utilization"] = utilization
    return out


def normalize_rows(path: Path, args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return grouped

        fields = reader.fieldnames
        time_col = first_existing(fields, TIME_COLUMNS)
        iso_time_col = first_existing(fields, ISO_TIME_COLUMNS)
        session_col = first_existing(fields, SESSION_COLUMNS)
        seq_col = first_existing(fields, SEQ_COLUMNS)
        rssi_col = first_existing(fields, RSSI_COLUMNS)
        tx_time_col = first_existing(fields, TX_TIME_COLUMNS)
        payload_col = first_existing(fields, PAYLOAD_COLUMNS)
        packet_rate_col = first_existing(fields, PACKET_RATE_COLUMNS)

        if not time_col and not iso_time_col:
            raise ValueError(f"{path}: missing time column, expected one of {TIME_COLUMNS}")
        if not rssi_col:
            raise ValueError(f"{path}: missing RSSI column, expected one of {RSSI_COLUMNS}")

        for row in reader:
            if row.get("record") and row["record"].strip().upper() != "RX":
                continue

            time_s = parse_time_seconds(row, time_col, iso_time_col)
            rssi = to_float(row.get(rssi_col))
            if time_s is None or rssi is None:
                continue

            session_id = row.get(session_col, "") if session_col else ""
            session_id = session_id.strip() or path.stem
            item = {
                "time_s": time_s,
                "session_id": session_id,
                "seq": to_float(row.get(seq_col)) if seq_col else None,
                "tx_time_s": (to_float(row.get(tx_time_col)) / 1000.0) if tx_time_col and to_float(row.get(tx_time_col)) is not None else None,
                "rssi": rssi,
                "noise_floor": to_float(row.get("noise_floor_dbm")),
                "snr": to_float(row.get("snr_db")),
                "gap": to_float(row.get("gap")),
                "payload_size": to_float(row.get(payload_col)) if payload_col else None,
                "packet_rate": to_float(row.get(packet_rate_col)) if packet_rate_col else None,
                "channel": to_float(row.get("channel")),
                "sig_mode": to_float(row.get("sig_mode")),
                "mcs": to_float(row.get("mcs")),
                "rate": to_float(row.get("rate")),
            }
            grouped[session_id].append(item)

    for rows in grouped.values():
        rows.sort(key=lambda item: (item["time_s"], item["seq"] if item["seq"] is not None else -1))
        if rows:
            base = rows[0]["time_s"]
            for row in rows:
                row["time_s"] -= base

    return grouped


def extract_features_for_session(
    source_file: str,
    session_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if len(rows) < args.min_packets:
        return []

    output_rows: list[dict[str, Any]] = []
    max_time = rows[-1]["time_s"]
    last_start = max_time - args.window_size
    if args.include_labels:
        last_start = max_time - args.window_size - args.horizon
    if last_start < 0:
        return []

    start_idx = 0
    end_idx = 0
    future_start_idx = 0
    future_end_idx = 0
    window_start = 0.0

    while window_start <= last_start + 1e-9:
        window_end = window_start + args.window_size
        future_start = window_end
        future_end = future_start + args.horizon

        while start_idx < len(rows) and rows[start_idx]["time_s"] < window_start:
            start_idx += 1
        while end_idx < len(rows) and rows[end_idx]["time_s"] < window_end:
            end_idx += 1

        window_rows = rows[start_idx:end_idx]
        if len(window_rows) >= args.min_packets:
            feature_row = build_feature_row(
                source_file,
                session_id,
                window_rows,
                window_start,
                window_end,
                args,
            )

            if args.include_labels:
                while future_start_idx < len(rows) and rows[future_start_idx]["time_s"] < future_start:
                    future_start_idx += 1
                while future_end_idx < len(rows) and rows[future_end_idx]["time_s"] < future_end:
                    future_end_idx += 1
                future_rows = rows[future_start_idx:future_end_idx]
                _, _, future_loss, _, _, _ = packet_loss(future_rows)
                future_snr = [row["snr"] for row in future_rows if row.get("snr") is not None]
                feature_row["packet_loss_future_2s"] = future_loss
                feature_row["snr_future_mean_2s"] = mean(future_snr)

            output_rows.append(feature_row)

        window_start += args.step_size

    return output_rows


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        matches = glob.glob(item)
        if not matches:
            matches = [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.glob("*.csv")))
            elif path.suffix.lower() == ".csv":
                paths.append(path)
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(path)
    return unique_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create XGBoost-ready sliding-window feature CSV from ESP-NOW packet logs."
    )
    parser.add_argument("inputs", nargs="+", help="Input CSV file(s), glob(s), or folder(s).")
    parser.add_argument("-o", "--output", default="xgboost_features.csv", help="Output feature CSV.")
    parser.add_argument("--window-size", type=float, default=2.0, help="Input window size in seconds.")
    parser.add_argument("--step-size", type=float, default=0.5, help="Sliding step size in seconds.")
    parser.add_argument("--horizon", type=float, default=2.0, help="Future label horizon in seconds.")
    parser.add_argument("--min-packets", type=int, default=5, help="Minimum received packets per window.")
    parser.add_argument("--packet-rate", type=float, default=20.0, help="Fallback packet rate if it cannot be inferred.")
    parser.add_argument("--payload-size", type=float, default=28.0, help="Fallback payload size in bytes if absent.")
    parser.add_argument("--rssi-bin-width", type=float, default=1.0, help="RSSI histogram bin width for entropy.")
    parser.add_argument("--bandwidth-hz", type=float, default=None, help="Optional bandwidth for Shannon-Hartley features.")
    parser.add_argument("--capacity-efficiency", type=float, default=1.0, help="Efficiency factor for effective capacity.")
    parser.add_argument("--include-labels", action="store_true", help="Append future Packet Loss/SNR labels for training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_paths = expand_inputs(args.inputs)
    if not input_paths:
        raise SystemExit("No CSV input files found.")

    fieldnames = FEATURE_COLUMNS + (LABEL_COLUMNS if args.include_labels else [])
    rows_written = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for path in input_paths:
            grouped = normalize_rows(path, args)
            for session_id, rows in grouped.items():
                feature_rows = extract_features_for_session(
                    path.name,
                    session_id,
                    rows,
                    args,
                )
                for row in feature_rows:
                    writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
                rows_written += len(feature_rows)

    print(f"Wrote {rows_written} feature rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
