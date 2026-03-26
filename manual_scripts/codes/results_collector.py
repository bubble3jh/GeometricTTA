"""
results_collector.py
Append per-step and per-run results to shared CSV files for cross-experiment analysis.

Usage:
    from results_collector import ResultsCollector
    rc = ResultsCollector(experiment="inst36g", run_id="k10_lam2.5",
                          K=10, dataset="cifar10_c", corruption="gaussian_noise",
                          severity=5, optimizer="AdamW", lr=0.001, wd=0.01,
                          n_steps=50, batch_size=200, lam=2.5, c_min=None)
    # inside loop:
    rc.log_step(step=1, online_acc=0.5, cat_pct=0.3, H_pbar=2.2, mean_ent=1.8)
    # after loop:
    rc.log_summary(final_online_acc=0.67, offline_acc=None, mf_gap=0.4)
"""

import csv
import os
import time
from pathlib import Path

# Shared results directory — lives under experiments/runs/
_RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "runs" / "lambda_results"
STEPS_CSV  = _RESULTS_DIR / "hp_steps.csv"
SUMMARY_CSV = _RESULTS_DIR / "hp_summary.csv"

_STEPS_FIELDS = [
    "timestamp_run", "experiment", "run_id", "K", "dataset", "corruption", "lam",
    "step", "n_steps", "online_acc", "cat_pct", "H_pbar", "mean_ent", "I_batch",
]
_SUMMARY_FIELDS = [
    "timestamp", "experiment", "run_id", "K", "dataset", "corruption", "severity",
    "optimizer", "lr", "wd", "n_steps", "batch_size", "lam", "c_min",
    "final_online_acc", "offline_acc", "mf_gap", "H_pbar_final", "I_batch_final", "cat_pct_final",
]


def _ensure_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def _append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    _ensure_csv(path, fieldnames)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writerow(row)


class ResultsCollector:
    def __init__(self, *, experiment: str, run_id: str, K: int, dataset: str,
                 corruption: str, severity: int = 5,
                 optimizer: str, lr: float, wd: float,
                 n_steps: int, batch_size: int = 200,
                 lam: float, c_min):
        self.meta = dict(
            experiment=experiment, run_id=run_id, K=K, dataset=dataset,
            corruption=corruption, severity=severity,
            optimizer=optimizer, lr=lr, wd=wd,
            n_steps=n_steps, batch_size=batch_size,
            lam=lam, c_min=c_min if c_min is not None else "",
        )
        self._ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._H_pbar_final = None
        self._mean_ent_final = None
        self._cat_pct_final = None

    def log_step(self, *, step: int, online_acc: float,
                 cat_pct: float, H_pbar: float, mean_ent: float) -> None:
        i_batch = H_pbar - mean_ent
        self._H_pbar_final = H_pbar
        self._mean_ent_final = mean_ent
        self._cat_pct_final = cat_pct
        row = dict(
            timestamp_run=self._ts,
            experiment=self.meta["experiment"],
            run_id=self.meta["run_id"],
            K=self.meta["K"],
            dataset=self.meta["dataset"],
            corruption=self.meta["corruption"],
            lam=f"{self.meta['lam']:.4f}",
            step=step,
            n_steps=self.meta["n_steps"],
            online_acc=f"{online_acc:.4f}",
            cat_pct=f"{cat_pct:.4f}",
            H_pbar=f"{H_pbar:.4f}",
            mean_ent=f"{mean_ent:.4f}",
            I_batch=f"{i_batch:.4f}",
        )
        _append_row(STEPS_CSV, _STEPS_FIELDS, row)

    def log_summary(self, *, final_online_acc: float, offline_acc=None,
                    mf_gap: float = None) -> None:
        H_pbar_f = self._H_pbar_final or 0.0
        mean_ent_f = self._mean_ent_final or 0.0
        I_batch_f = H_pbar_f - mean_ent_f
        row = {**self.meta,
               "timestamp": self._ts,
               "final_online_acc": f"{final_online_acc:.4f}",
               "offline_acc": f"{offline_acc:.4f}" if offline_acc is not None else "",
               "mf_gap": f"{mf_gap:.4f}" if mf_gap is not None else "",
               "H_pbar_final": f"{H_pbar_f:.4f}",
               "I_batch_final": f"{I_batch_f:.4f}",
               "cat_pct_final": f"{self._cat_pct_final:.4f}" if self._cat_pct_final is not None else "",
               }
        _append_row(SUMMARY_CSV, _SUMMARY_FIELDS, row)
