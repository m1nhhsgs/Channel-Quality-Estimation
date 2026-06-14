#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None


RX_FIELDS = [
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

SESSION_FIELDS = [
    "session_id",
    "start_time",
    "location",
    "distance_m",
    "link_type",
    "obstacle",
    "send_interval_ms",
    "channel",
    "notes",
]


def prompt_value(label, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{label}{suffix}: ").strip()
    if value == "" and default is not None:
        return str(default)
    return value


def append_csv_row(path, fieldnames, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def list_serial_ports():
    if list_ports is None:
        return []
    return list(list_ports.comports())


def choose_port(port_arg):
    if port_arg:
        return port_arg

    ports = list_serial_ports()
    if ports:
        print("Available serial ports:")
        for idx, port in enumerate(ports, start=1):
            print(f"  {idx}. {port.device} - {port.description}")
        selected = prompt_value("Port number or port name")
        if selected.isdigit():
            index = int(selected) - 1
            if 0 <= index < len(ports):
                return ports[index].device
        return selected

    return prompt_value("Serial port, example /dev/cu.usbserial-0001 or COM5")


def build_session(args):
    now = datetime.now().isoformat(timespec="seconds")
    session_id = args.session_id or prompt_value("session_id", "run01")

    return {
        "session_id": session_id,
        "start_time": now,
        "location": args.location or prompt_value("location", "lab_room"),
        "distance_m": args.distance_m or prompt_value("distance_m", "1.0"),
        "link_type": args.link_type or prompt_value("link_type (LOS/NLOS)", "LOS"),
        "obstacle": args.obstacle or prompt_value("obstacle", "none"),
        "send_interval_ms": args.send_interval_ms or prompt_value("send_interval_ms", "50"),
        "channel": args.channel or prompt_value("channel", "1"),
        "notes": args.notes if args.notes is not None else prompt_value("notes", ""),
    }


def parse_rx_line(line):
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(RX_FIELDS):
        return None
    return dict(zip(RX_FIELDS, parts))


def main():
    parser = argparse.ArgumentParser(
        description="Log ESP-NOW receiver CSV and session metadata."
    )
    parser.add_argument("--port", help="Serial port, for example /dev/cu.usbserial-0001 or COM5")
    parser.add_argument("--baud", default=115200, type=int)
    parser.add_argument("--session-id")
    parser.add_argument("--location")
    parser.add_argument("--distance-m")
    parser.add_argument("--link-type", choices=["LOS", "NLOS"])
    parser.add_argument("--obstacle")
    parser.add_argument("--send-interval-ms")
    parser.add_argument("--channel")
    parser.add_argument("--notes")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-file", help="Override output CSV path. Defaults to data/raw/<session_id>.csv.")
    parser.add_argument(
        "--expected-packets",
        type=int,
        default=0,
        help="Stop after saving this many valid RX packets. Use 0 to run until Ctrl+C.",
    )
    args = parser.parse_args()

    if serial is None:
        raise SystemExit(
            "Missing pyserial. Install it with: python3 -m pip install pyserial"
        )

    port = choose_port(args.port)
    session = build_session(args)

    sessions_path = os.path.join(args.data_dir, "sessions.csv")
    receiver_path = args.output_file or os.path.join(
        args.data_dir, "raw", f"{session['session_id']}.csv"
    )

    append_csv_row(sessions_path, SESSION_FIELDS, session)

    receiver_fields = ["pc_timestamp"] + RX_FIELDS
    print()
    print(f"Session saved: {sessions_path}")
    print(f"Receiver log:  {receiver_path}")
    if args.expected_packets > 0:
        print(f"Auto stop after: {args.expected_packets} valid RX packets")
    print("Press Ctrl+C to stop logging.")
    print()

    saved_packets = 0
    with serial.Serial(port, args.baud, timeout=1) as ser:
        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            print(line)

            if not line.startswith("RX,"):
                continue

            row = parse_rx_line(line)
            if row is None:
                continue

            if row["session_id"] != session["session_id"]:
                row["session_id"] = session["session_id"]

            row = {
                "pc_timestamp": datetime.now().isoformat(timespec="seconds"),
                **row,
            }
            append_csv_row(receiver_path, receiver_fields, row)
            saved_packets += 1

            if args.expected_packets > 0 and saved_packets >= args.expected_packets:
                print()
                print(f"Reached expected packets: {saved_packets}")
                print("Logging stopped.")
                break


if __name__ == "__main__":
    main()
