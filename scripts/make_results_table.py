#!/usr/bin/env python3
"""Regenerate the paper's results table from runs/*.json.

Usage:
    PYTHONPATH=src python scripts/make_results_table.py [runs_dir] [--markdown]
"""

import argparse
import glob
import json
import os

PROBE_ORDER = [
    "picture_vocabulary", "counting", "localization_h", "localization_v",
    "who_has_more", "spatial_details", "memory", "vtwt_phrase", "vtwt_image",
    "caption",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir", nargs="?", default="runs")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    rows = {}
    for f in sorted(glob.glob(os.path.join(args.runs_dir, "*.json"))):
        name = os.path.basename(f)[:-5]
        rows[name] = json.load(open(f))

    if args.markdown:
        header = "| method | " + " | ".join(PROBE_ORDER) + " | overall |"
        print(header)
        print("|---" * (len(PROBE_ORDER) + 2) + "|")
        for name, r in rows.items():
            vals = []
            for p in PROBE_ORDER:
                v = r.get(p, {})
                vals.append(f"{v.get('acc', v.get('f1', 0.0)):.3f}")
            print(f"| {name} | " + " | ".join(vals) +
                  f" | {r.get('overall_choice_acc', 0.0):.3f} |")
    else:
        print("method".ljust(14) + "".join(p.ljust(20) for p in PROBE_ORDER))
        for name, r in rows.items():
            row = name.ljust(14)
            for p in PROBE_ORDER:
                v = r.get(p, {})
                row += f"{v.get('acc', v.get('f1', 0.0)):<20.3f}"
            print(row)
        print()
        for name, r in rows.items():
            a = r.get("analysis", {})
            if a:
                print(f"{name}: loc_err={a['mean_localization_error_patch']:.2f}"
                      f" gate_content={a['gate_content']:.3f}"
                      f" gate_function={a['gate_function']:.3f}")


if __name__ == "__main__":
    main()
