---
description: Distributed experiment protocol — PC (RTX 3070 Ti) + Laptop (RTX 4060, SSH via Tailscale). Always applies.
---

# Distributed Experiment Protocol

## Infrastructure

| Machine | GPU | Role |
|---------|-----|------|
| PC | RTX 3070 Ti | Primary / Orchestrator (Claude Code runs here) |
| Laptop | RTX 4060 Laptop | Secondary / Remote Worker |

Laptop access: `ssh -p 2222 jino@100.125.103.5` (Tailscale IP, stable across reboots)
Remote PATH: always prepend `source /home/jino/miniconda3/etc/profile.d/conda.sh &&` to SSH commands.
(`source ~/.bashrc` does NOT activate conda in non-interactive shells — use conda.sh directly.)

---

## Step 0: SSH Connectivity Check (automatic before any laptop work)
```bash
ssh -p 2222 jino@100.125.103.5 "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader" 2>&1
```
- PASS → proceed.
- FAIL → notify user: "⚠️ 노트북 SSH 연결 실패. 노트북 WSL에서 `sudo service ssh start` 실행 필요합니다." Do NOT proceed with laptop work until confirmed.

---

## Step 1: Execution Mode — ALWAYS ask, NEVER skip

**모든 실험 시작 전에 반드시 AskUserQuestion으로 실행 위치를 확인해야 함. 기본값 가정 금지.**

Options to present:
1. PC + Laptop 분산 처리 — 작업을 양쪽에 분배
2. PC 단독 처리 — PC에서만 실행
3. Laptop 단독 처리 — 노트북에서만 실행

---

## Step 2: Code Sync (PC → Laptop, automatic when laptop is involved)
```bash
rsync -avz --exclude '.git' \
           --exclude '__pycache__' \
           --exclude '*.pt' \
           --exclude '*.pth' \
           --exclude '*.tar.gz' \
           --exclude 'experiments/baselines/BATCLIP/classification/data/' \
           --exclude 'experiments/CALM/data' \
           --exclude 'experiments/runs/' \
           --exclude 'wandb/' \
           --exclude 'cookies.json' \
           -e "ssh -p 2222" \
           ~/Lab/v2/ jino@100.125.103.5:~/Lab/v2/
```

**DATA PATH SAFETY**: On PC, `experiments/CALM/data` is a **symlink** → rsyncing it would overwrite laptop's real data with a broken symlink. The exclude above prevents this. Do NOT remove it.

---

## Step 3: Remote Launch Pattern (nohup + PID capture)
```bash
# Launch on laptop, capture PID
LAPTOP_PID=$(ssh -p 2222 jino@100.125.103.5 \
  "source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && \
   cd ~/Lab/v2 && nohup python <script> <args> > <logfile> 2>&1 & echo \$!")
echo "Laptop PID: $LAPTOP_PID"
```

For PC-side background launch:
```bash
cd ~/Lab/v2 && nohup python <script> <args> > <logfile> 2>&1 &
PC_PID=$!
echo "PC PID: $PC_PID"
```

---

## Step 4: Distributed Announcement Format
After launching on both machines:
```
분산 실험 시작.
  PC     PID: <PC_PID>    로그: tail -f <pc_logfile>
  Laptop PID: <LAPTOP_PID> 로그: ssh -p 2222 jino@100.125.103.5 "tail -f <laptop_logfile>"
실시간 모니터 (PC 측): python manual_scripts/codes/monitor.py
```

---

## Step 5: Result Collection (Laptop → PC, automatic after completion)
```bash
rsync -avz -e "ssh -p 2222" \
    jino@100.125.103.5:~/Lab/v2/experiments/runs/<results_path>/ \
    ~/Lab/v2/experiments/runs/<results_path>/
```

---

## Remote Health Check
```bash
# Check if laptop process is alive
ssh -p 2222 jino@100.125.103.5 "ps -p <LAPTOP_PID> && tail -n 20 <laptop_logfile>"
```

---

## Troubleshooting
- **SSH timeout after laptop reboot**: WSL IP changed. User must run on laptop Windows (Admin PowerShell):
  ```powershell
  wsl hostname -I  # get new WSL IP
  netsh interface portproxy delete v4tov4 listenport=2222 listenaddress=0.0.0.0
  netsh interface portproxy add v4tov4 listenport=2222 listenaddress=0.0.0.0 connectport=22 connectaddress=<NEW_IP>
  ```
  Tailscale IP `100.125.103.5` stays stable — only internal WSL IP changes.
