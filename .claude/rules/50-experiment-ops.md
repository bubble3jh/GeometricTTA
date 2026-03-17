---
description: Experiment operations — launch checklist, monitor, status writer (always applies, regardless of CWD)
---

# Experiment operations

## Pre-launch checklist (mandatory)
Before starting any background experiment:
```bash
ps aux | grep "python.*run_" | grep -v grep   # zombie 확인
nvidia-smi                                     # GPU 점유 확인
# VRAM free < 4 GB → 실행 금지
```

## Run announcement (mandatory)
Background 실험 시작 직후 반드시 다음 형식으로 안내:
```
GPU 체크 완료. 실험을 시작합니다. (PID: XXXXX)
로그: tail -f <log_path>
실시간 모니터: python manual_scripts/codes/monitor.py
```
→ `/tmp/exp_status.json`을 1초마다 읽어 Rich UI로 phase/step/online_acc/ETA 표시.

## Status writer (mandatory for new scripts)
모든 새 실험 스크립트는 `status_writer.py`를 사용해야 함:
```python
from status_writer import write_status, compute_eta
# _adapt_loop 안, step 로깅 직후
write_status(
    script=os.path.basename(__file__),
    phase=phase, phase_total=phase_total,
    corruption=corruption, corr_idx=corr_idx, corr_total=corr_total,
    step=step+1, n_steps=n_steps,
    online_acc=online_acc, s_per_step=s_per_step,
    eta=compute_eta(step+1, n_steps, corr_idx, corr_total, s_per_step),
)
```

## Bash wrapper exit-code + PID standard (mandatory for new .sh wrappers)
모든 새 `.sh` 래퍼는 반드시 Python 종료 코드를 전파하고 PID를 기록해야 함:

```bash
"${PYTHON}" "${SCRIPT}" "${ARGS}" 2>&1 | tee "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}
echo "[INFO] Exit code: $EXIT_CODE. Log: $LOGFILE"
exit $EXIT_CODE
```

**중요:**
- `ps -p <bash_pid>`는 Python child가 죽어도 bash wrapper가 살아있으면 계속 alive로 보임.
- Python child PID를 별도로 기록하거나, log에서 완료 sentinel을 확인하는 것이 신뢰할 수 있는 완료 감지 방법.
- 권장: `ps aux | grep "python.*run_"` + log tail로 이중 확인.

## Block-structured script 복구 (mandatory for scripts > 30 min)
Block A/B/C 구조를 가진 긴 스크립트는 반드시:
1. 각 block 완료 후 즉시 중간 결과를 JSON으로 저장
2. `build_summary` / `write_report`는 이 JSON들을 읽는 독립 단계로 구현
3. `--block report` 옵션으로 GPU 없이 요약/리포트 재생성 가능하게 설계

복구 패턴: 실험 loop 완료 후 summary/report 단계에서 crash 시 → 저장된 JSON을 읽어
`build_summary` + `write_report`를 직접 호출해 GPU 재실행 없이 복구 가능.

## Note on CLAUDE.md scoping
`experiments/CLAUDE.md`의 상세 규칙은 CWD가 `experiments/` 하위일 때만 로드됨.
실험 launch는 root session에서 발생하므로 이 파일(`.claude/rules/50-experiment-ops.md`)이 항상 적용되는 단일 진실 소스.
