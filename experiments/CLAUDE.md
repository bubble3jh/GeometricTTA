# Experiments rules

## Hard rules (never violate)
1. Every run reproducible from a single command (or make target). Prefer config-driven code (yaml/json/argparse) over hard-coded constants.
2. Never run multiple CUDA experiments in parallel (RTX 3070 Ti, 8 GB VRAM). Each ViT-B-16 run needs ~2–3 GB (more for multiview n_aug=5). Sequential only.
3. If VRAM free < 4 GB, do not start a new run.
4. Never add follow-up diagnostics while main experiment is still running.
5. Never merge code that changes results without a baseline comparison + delta explanation.
6. Fixed config — do NOT change unless user explicitly says so:
   - open_clip 2.20.0 (QuickGELU), seed=1, N=10000, gaussian_noise only, severity=5
   - BATCLIP baseline: **acc=60.60%** (paper 61.13%; ~0.5 pp GPU hardware gap)

## Pre-experiment checklist
```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits
free -h
nvidia-smi            # no competing GPU processes
```

## Run logging
Write to `experiments/runs/<yyyymmdd>/<exp_name>/`. Every run must contain:
- `command.txt`, `config.yaml`, `metrics.jsonl` (or `.csv`), `summary.md`
- Recorded: git hash, full config, seed(s), python+cuda versions
- Optional: `wandb_url.txt`

## Verification
After any code change affecting training/eval:
1. Run smallest fast check (unit test / smoke run) first.
2. Then run the intended experiment.

## Codebase patterns (reference)
- Methods: `TTAMethod` subclasses via `@ADAPTATION_REGISTRY.register()`; key = lowercase class name.
- Forward: `logits, img_feat, text_feat, img_pre, text_pre = model(imgs, return_features=True)`
  - `img_feat`, `text_feat` = L2-normalized. All outputs on **GPU**.
- Configs: YACS-based; add new sections to `conf.py` before `_CFG_DEFAULT = _C.clone()`.
- Runner: add `BATCLIP/classification/` to `sys.path`, then import `conf`, `models.model`, `datasets.data_loading`.
- BATCLIP loss = `L_ent - L_i2t - L_inter_mean` with LayerNorm adaptation (not pure zero-shot).

## GPU ↔ CPU device rules
- **Model forward returns GPU tensors always.** `.float()` / `.detach()` do NOT move to CPU.
- **collect_all_features**: append with `.float().cpu()` — omitting `.cpu()` → OOM after ~50 batches.
- **Diagnostic functions**: normalize device at entry (`tensor.cpu().float()`) before any cross-source matmul.
- **Between runs**: `del` tensors + `torch.cuda.empty_cache()` before next `collect_all_features`.
- **results_combined**: store only Python scalars/lists (`.item()`, `.tolist()`), never tensors.
- **model_state_init**: `deepcopy(state_dict())` stays on GPU; peak ~1 GB — do not add another deepcopy.

## Run announcement (required)
실험을 background로 시작할 때마다 반드시 다음 형식으로 안내:
```
GPU 체크 완료. 수행을 시작합니다.
결과를 추적하시려면: python manual_scripts/codes/monitor.py
로그 확인: tail -f <log_path>
```

## Status writer (required for new scripts)
모든 새 실험 스크립트는 `status_writer.py`를 import해서 매 DIAG_INTERVAL step마다 상태를 기록해야 함.

```python
# script 상단
from status_writer import write_status, compute_eta

# _adapt_loop 또는 step 로깅 직후
if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
    s_per_step = elapsed_so_far / (step + 1)
    write_status(
        script      = os.path.basename(__file__),
        phase       = current_phase, phase_total = total_phases,
        corruption  = corruption,    corr_idx    = corr_idx, corr_total = corr_total,
        step        = step + 1,      n_steps     = n_steps,
        online_acc  = online_acc,
        s_per_step  = s_per_step,
        eta         = compute_eta(step+1, n_steps, corr_idx, corr_total, s_per_step),
    )
```

모니터 실행: `python manual_scripts/codes/monitor.py` (별도 터미널)

## Multi-phase memory checklist
When a script runs 3+ sequential phases:
1. **main() scope leaks** — `del` intermediate CPU tensors before next phase.
2. **results_combined audit** — assert no `torch.Tensor` / `np.ndarray` values after each run.
3. **Optimizer state** — create fresh `AdamW` per run; never hoist to main() scope.
4. **Activation graph** — store loss as `float(loss.item())`, not as tensor.
5. **del ordering** — delete BEFORE next `collect_all_features`, not after.