#!/usr/bin/env python3
"""
Split labeled feature data for train/test.

Two modes are supported:
1. Hold out scenario_id values from one combined dataset.
2. Hold out one whole labeled CSV file from a folder, using all other files for train.

Example:
    python split_dataset.py data/ml_dataset.csv --test-scenario-id run10
    python split_dataset.py data/ml_dataset.csv --test-scenario-id run13,run14
    python split_dataset.py data/labeled --test-file RB2_run14_labeled.csv

Result:
    data/train_dataset.csv
    data/test_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split ESP-NOW dataset by scenario_id or by holding out one labeled CSV file."
    )
    parser.add_argument("dataset", help="Input labeled dataset CSV or folder of labeled CSV files.")
    parser.add_argument(
        "--test-scenario-id",
        action="append",
        help="Scenario/run kept for test. Can be repeated or comma-separated, e.g. run13,run14.",
    )
    parser.add_argument("--test-file", help="CSV file name/path kept for test when dataset is a folder.")
    parser.add_argument("--file-pattern", default="*_labeled.csv", help="File pattern used in folder split mode.")
    parser.add_argument("--train-output", default="data/train_dataset.csv", help="Output train CSV.")
    parser.add_argument("--test-output", default="data/test_dataset.csv", help="Output test CSV.")
    parser.add_argument("--group-column", default="scenario_id", help="Scenario ID column.")
    parser.add_argument("--target-column", default="packet_loss_future_2s", help="Required label column.")
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="Print available scenario_id values and exit without splitting.",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="Print available labeled CSV files and exit without splitting.",
    )
    return parser.parse_args()


def parse_test_scenario_ids(values: list[str] | None) -> set[str]:
    if not values:
        return set()

    aliases = {
        "RB2_run13+14": {"run13", "run14"},
        "RB2_run13_14": {"run13", "run14"},
    }
    scenario_ids: set[str] = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            scenario_ids.update(aliases.get(item, {item}))
    return scenario_ids


def read_labeled_rows(
    dataset_path: Path,
    group_column: str,
    target_column: str,
) -> tuple[list[str], list[dict[str, str]]]:
    with dataset_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header.")
        fieldnames = reader.fieldnames
        if group_column not in fieldnames:
            raise SystemExit(f"Missing group column '{group_column}'.")
        if target_column not in fieldnames:
            raise SystemExit(f"Missing target column '{target_column}'. Recreate dataset with --include-labels.")
        rows = [row for row in reader if row.get(target_column, "").strip() != ""]

    if not rows:
        raise SystemExit("No labeled rows found.")
    return fieldnames, rows


def read_labeled_file(dataset_path: Path, target_column: str) -> tuple[list[str], list[dict[str, str]]]:
    with dataset_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"{dataset_path}: input CSV has no header.")
        fieldnames = list(reader.fieldnames)
        if target_column not in fieldnames:
            raise SystemExit(f"{dataset_path}: missing target column '{target_column}'.")
        rows = [row for row in reader if row.get(target_column, "").strip() != ""]
    return fieldnames, rows


def list_labeled_files(folder: Path, pattern: str, target_column: str) -> list[Path]:
    files = sorted(path for path in folder.glob(pattern) if path.is_file())
    usable_files: list[Path] = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and target_column in reader.fieldnames:
                usable_files.append(path)
    return usable_files


def split_by_file(args: argparse.Namespace) -> int:
    folder = Path(args.dataset)
    if not folder.is_dir():
        raise SystemExit("--test-file mode requires dataset to be a folder.")

    files = list_labeled_files(folder, args.file_pattern, args.target_column)
    if args.list_files:
        print(f"Available labeled files in {folder}:")
        for path in files:
            _, rows = read_labeled_file(path, args.target_column)
            print(f"  {path.name}: {len(rows)} rows")
        return 0

    if not args.test_file:
        raise SystemExit("Missing --test-file. Use --list-files to see available files.")

    requested = Path(args.test_file).name
    test_files = [path for path in files if path.name == requested]
    if not test_files:
        print(f"Available labeled files in {folder}:")
        for path in files:
            print(f"  {path.name}")
        raise SystemExit(f"Test file '{requested}' was not found.")

    test_file = test_files[0]
    train_files = [path for path in files if path != test_file]
    if not train_files:
        raise SystemExit("Train set is empty. Need at least two labeled files.")

    fieldnames, test_rows = read_labeled_file(test_file, args.target_column)
    train_rows: list[dict[str, str]] = []
    for path in train_files:
        current_fieldnames, rows = read_labeled_file(path, args.target_column)
        if current_fieldnames != fieldnames:
            raise SystemExit(f"{path}: header differs from {test_file}.")
        train_rows.extend(rows)

    write_csv(Path(args.train_output), fieldnames, train_rows)
    write_csv(Path(args.test_output), fieldnames, test_rows)

    print("Split method: holdout_file")
    print(f"Test file: {test_file.name}")
    print(f"Train files: {len(train_files)}")
    print(f"Train rows: {len(train_rows)} -> {args.train_output}")
    print(f"Test rows:  {len(test_rows)} -> {args.test_output}")
    return 0


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_scenarios(rows: list[dict[str, str]], group_column: str) -> None:
    counts = Counter(row[group_column] for row in rows)
    print(f"Available {group_column} values:")
    for scenario_id, count in sorted(counts.items()):
        print(f"  {scenario_id}: {count} rows")


def main() -> int:
    args = parse_args()
    if args.test_file or args.list_files:
        return split_by_file(args)

    fieldnames, rows = read_labeled_rows(
        Path(args.dataset),
        args.group_column,
        args.target_column,
    )

    if args.list_scenarios:
        print_scenarios(rows, args.group_column)
        return 0

    test_scenario_ids = parse_test_scenario_ids(args.test_scenario_id)
    if not test_scenario_ids:
        raise SystemExit("Missing --test-scenario-id. Use --list-scenarios to see available IDs.")

    scenario_counts = Counter(row[args.group_column] for row in rows)
    missing_scenarios = sorted(test_scenario_ids - set(scenario_counts))
    if missing_scenarios:
        print_scenarios(rows, args.group_column)
        raise SystemExit(f"Scenario(s) not found: {', '.join(missing_scenarios)}")

    train_rows = [row for row in rows if row[args.group_column] not in test_scenario_ids]
    test_rows = [row for row in rows if row[args.group_column] in test_scenario_ids]

    if not train_rows:
        raise SystemExit("Train set is empty. Dataset must contain at least two scenario_id values.")
    if not test_rows:
        raise SystemExit("Test set is empty.")

    write_csv(Path(args.train_output), fieldnames, train_rows)
    write_csv(Path(args.test_output), fieldnames, test_rows)

    print(f"Split method: holdout_scenario_id")
    print(f"Test {args.group_column}: {', '.join(sorted(test_scenario_ids))}")
    print(f"Train rows: {len(train_rows)} -> {args.train_output}")
    print(f"Test rows:  {len(test_rows)} -> {args.test_output}")
    print(f"Train scenarios: {len(scenario_counts) - len(test_scenario_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
