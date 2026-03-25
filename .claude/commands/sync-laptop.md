# sync-laptop

Distributed experiment helper for PC ↔ Laptop (RTX 4060, Tailscale 100.125.103.5:2222).

## Usage

### `/sync-laptop check`
Verify SSH connectivity and GPU status on laptop:
```bash
ssh -p 2222 jino@100.125.103.5 "source ~/.bashrc && nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader && ps aux | grep 'python.*run_' | grep -v grep"
```

### `/sync-laptop code`
Sync code from PC → Laptop (excludes data, runs, model weights):
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

### `/sync-laptop results <path>`
Pull results from Laptop → PC:
```bash
rsync -avz -e "ssh -p 2222" \
    jino@100.125.103.5:~/Lab/v2/<path>/ \
    ~/Lab/v2/<path>/
```
Replace `<path>` with the experiment results directory (e.g., `experiments/runs/h2_sweep/run_20260316_123456`).
