#!/usr/bin/env python3
"""
merge_inst33_k1000.py
=====================
Merge K=1000 imagenet_c results from two separate output dirs (PC + laptop)
into a single combined summary.

Usage:
    python manual_scripts/codes/merge_inst33_k1000.py \
        --laptop-dir outputs/inst33_k1000 \
        --pc-dir     outputs/inst33_k1000_pc \
        --out-dir    outputs/inst33_k1000_merged

The script reads per-corruption JSON files from both dirs, checks for overlap,
and writes a merged summary.txt + k1000_results.csv.
"""

import argparse
import csv
import json
import os
import sys

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
    "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def load_jsons(main_table_dir):
    results = {}
    for corr in ALL_CORRUPTIONS:
        p = os.path.join(main_table_dir, f"{corr}.json")
        if os.path.exists(p):
            with open(p) as f:
                results[corr] = json.load(f)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop-dir", required=True, help="outputs/inst33_k1000 on laptop (rsynced to PC)")
    ap.add_argument("--pc-dir",     required=True, help="outputs/inst33_k1000_pc on PC")
    ap.add_argument("--out-dir",    required=True, help="merged output dir")
    args = ap.parse_args()

    laptop_main = os.path.join(args.laptop_dir, "main_table")
    pc_main     = os.path.join(args.pc_dir,     "main_table")
    out_main    = os.path.join(args.out_dir,    "main_table")
    os.makedirs(out_main, exist_ok=True)

    laptop_results = load_jsons(laptop_main)
    pc_results     = load_jsons(pc_main)

    # overlap check
    overlap = set(laptop_results) & set(pc_results)
    if overlap:
        print(f"ERROR: overlapping corruptions between PC and laptop: {overlap}", file=sys.stderr)
        sys.exit(1)

    # missing check
    combined = {**laptop_results, **pc_results}
    missing = [c for c in ALL_CORRUPTIONS if c not in combined]
    if missing:
        print(f"WARNING: missing corruptions (not yet done): {missing}")

    # merge in canonical order
    all_results = []
    for corr in ALL_CORRUPTIONS:
        if corr in combined:
            all_results.append(combined[corr])

    # write merged JSONs
    for r in all_results:
        out_json = os.path.join(out_main, f"{r['corruption']}.json")
        with open(out_json, "w") as f:
            json.dump(r, f, indent=2)

    # write CSV
    fields = ["corruption", "lambda_0", "lambda_eff", "online_acc", "offline_acc", "timestamp"]
    csv_path = os.path.join(out_main, "k1000_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)

    # write summary
    accs = [r["offline_acc"] for r in all_results if not r.get("killed")]
    summary_path = os.path.join(out_main, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"dataset: imagenet_c  K=1000\n")
        f.write(f"n_corruptions: {len(all_results)}\n")
        if accs:
            f.write(f"mean_offline_acc: {sum(accs)/len(accs):.5f}\n")
        for r in all_results:
            src = "pc" if r["corruption"] in pc_results else "laptop"
            f.write(f"  {r['corruption']:22s} offline={r['offline_acc']:.5f}"
                    f"  λ_eff={r['lambda_eff']:.4f}  [{src}]\n")

    print(f"Merged {len(all_results)}/15 corruptions  "
          f"(laptop={len(laptop_results)}, pc={len(pc_results)})")
    if accs:
        print(f"mean_offline_acc: {sum(accs)/len(accs):.5f}")
    print(f"Saved: {summary_path}")
    if missing:
        print(f"Still missing: {missing}")


if __name__ == "__main__":
    main()
