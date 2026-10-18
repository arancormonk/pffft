#!/usr/bin/env python3
"""Summarize hotloops CSVs without dependencies or pooled-size averages."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fields = ["suite", "build", "precision", "backend", "kind", "operation", "n", "unit",
              "samples", "median", "min", "max", "mad_percent"]
    with args.output.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for path in args.inputs:
            groups = defaultdict(list)
            with path.open(newline="") as source:
                for row in csv.DictReader(source):
                    unit = "ns_per_call" if "ns_per_call" in row else "ns_per_sample"
                    key = tuple(row.get(f, "") for f in fields[1:7]) + (unit,)
                    groups[key].append(float(row[unit]))
            for key, values in sorted(groups.items()):
                mid = median(values)
                row = dict(zip(fields[1:8], key))
                row.update(suite=path.stem, samples=len(values), median=f"{mid:.6f}",
                           min=f"{min(values):.6f}", max=f"{max(values):.6f}",
                           mad_percent=f"{100 * median(abs(v - mid) for v in values) / mid:.4f}")
                writer.writerow(row)


if __name__ == "__main__":
    main()
