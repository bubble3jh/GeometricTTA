#!/usr/bin/env python3
"""
Update notes/lossB_auto_results.csv with latest experiment data.
Run anytime to refresh: python manual_scripts/codes/update_results_csv.py
Automatically pulls latest 2-point grid JSONs from per_corr_grid/k100/run_*/
"""
import json, csv, glob, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/runs"
OUT  = REPO / "notes/lossB_auto_results.csv"

ALL_CORRUPTIONS = [
    "gaussian_noise","shot_noise","impulse_noise",
    "defocus_blur","glass_blur","motion_blur","zoom_blur",
    "snow","frost","fog",
    "brightness","contrast","elastic_transform",
    "pixelate","jpeg_compression",
]

FIELDS = [
    "K","corruption",
    "lambda_auto","c","c_negative","cos_angle",
    "lambda_low","lambda_high","I_batch_step0",
    "online_acc","offline_acc","cat_pct",
    "killed","elapsed_s",
    "grid_lam_low","grid_lam_high","grid_best_online","grid_best_offline",
]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_lossb(k):
    pattern = str(BASE / f"per_corr_grid/k{k}/lossB_auto_*/summary.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    # use latest
    return {r["corruption"]: r for r in load_json(files[-1])["per_corruption"]}


def load_phase3(k):
    pattern = str(BASE / f"admissible_interval/k{k}/run_*/phase3_summary.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    return {r["corruption"]: r for r in load_json(files[-1])["per_corruption"]}


def load_grid(k):
    pattern = str(BASE / f"per_corr_grid/k{k}/run_*/")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        return {}
    run_dir = dirs[-1]
    grid = {}
    for f in glob.glob(os.path.join(run_dir, "*.json")):
        r = load_json(f)
        grid.setdefault(r["corruption"], []).append(r)
    return grid


def build_rows(k):
    lossb  = load_lossb(k)
    phase3 = load_phase3(k)
    grid   = load_grid(k)
    rows = []
    for corr in ALL_CORRUPTIONS:
        lb = lossb.get(corr, {})
        p3 = phase3.get(corr, {})
        gr = sorted(grid.get(corr, []), key=lambda x: x.get("lam", 0))
        g_online  = [r["online_acc"]  for r in gr]
        g_offline = [r["offline_acc"] for r in gr]
        rows.append({
            "K": k, "corruption": corr,
            "lambda_auto":    round(lb.get("lambda_auto") or 0, 4),
            "c":              round(p3.get("c") or 0, 4),
            "c_negative":     p3.get("c_negative"),
            "cos_angle":      round(p3.get("cos_angle") or 0, 4),
            "lambda_low":     round(p3.get("lambda_low") or 0, 4),
            "lambda_high":    round(p3.get("lambda_high") or 0, 4),
            "I_batch_step0":  round(p3.get("I_batch") or 0, 4),
            "online_acc":     lb.get("online_acc"),
            "offline_acc":    lb.get("offline_acc"),
            "cat_pct":        lb.get("cat_pct"),
            "killed":         lb.get("killed"),
            "elapsed_s":      round(lb.get("elapsed_s") or 0, 1),
            "grid_lam_low":   gr[0]["lam"] if len(gr) >= 1 else None,
            "grid_lam_high":  gr[1]["lam"] if len(gr) >= 2 else None,
            "grid_best_online":  round(max(g_online),  4) if g_online  else None,
            "grid_best_offline": round(max(g_offline), 4) if g_offline else None,
        })
    return rows


def main():
    K_VALUES = [10, 100]  # add 1000 when available
    all_rows = []
    for k in K_VALUES:
        rows = build_rows(k)
        n_lossb = sum(1 for r in rows if r["online_acc"] is not None)
        n_grid  = sum(1 for r in rows if r["grid_best_online"] is not None)
        print(f"K={k}: lossB_auto={n_lossb}/15, 2pt_grid={n_grid}/15")
        all_rows.extend(rows)

    # Drop rows where lambda_auto == 2.0 (bug fallback — no valid gradient ratio)
    valid_rows = [r for r in all_rows if r["lambda_auto"] != 2.0]
    dropped = len(all_rows) - len(valid_rows)
    if dropped:
        print(f"Dropped {dropped} rows with lambda_auto=2.0 (bug fallback)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(valid_rows)
    all_rows = valid_rows
    print(f"\nUpdated: {OUT}  ({len(all_rows)} rows)")

    # summary stats
    print(f"\n{'K':>4} {'corruption':<22} {'λ_auto':>7} {'c':>7} {'cos':>6} "
          f"{'c<0':>5} {'I_b0':>6} {'online':>7} {'offline':>8} {'cat%':>6} {'grid_best':>10}")
    print("─" * 98)
    for r in all_rows:
        g  = f"{r['grid_best_online']:.4f}" if r["grid_best_online"] is not None else "  --  "
        cn = "True " if r["c_negative"] else "False"
        oa = f"{r['online_acc']:.4f}"  if r["online_acc"]  is not None else "  --  "
        fa = f"{r['offline_acc']:.4f}" if r["offline_acc"] is not None else "  --  "
        cp = f"{r['cat_pct']:.3f}"     if r["cat_pct"]     is not None else " -- "
        print(f"{r['K']:>4} {r['corruption']:<22} {r['lambda_auto']:>7.4f} {r['c']:>7.3f} "
              f"{r['cos_angle']:>6.3f} {cn:>5} {r['I_batch_step0']:>6.4f} "
              f"{oa:>7} {fa:>8} {cp:>6} {g:>10}")
        if r["corruption"] == "jpeg_compression":
            print("─" * 98)


if __name__ == "__main__":
    main()
