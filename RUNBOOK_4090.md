# RTX 4090 session runbook

Everything below runs on the rented machine, inside `tmux`, from the repo folder.
Times in brackets are estimates.

## 0. Rent the machine (Vast.ai)
- RTX 4090 x1, **CPU >= 16 cores**, disk >= 80 GB, download >= 500 Mbps,
  verified / high reliability, **on-demand** (not interruptible).
- Template: a PyTorch / CUDA 12.x image with SSH.

## 1. Connect and start tmux  [2 min]
```bash
ssh -p <PORT> root@<IP>          # command shown on the Vast instance card
tmux new -s main                 # detach: Ctrl-b then d   reattach: tmux attach -t main
```

## 2. Get the code  [2 min]
Either clone your repo:
```bash
git clone https://github.com/<you>/<repo>.git repo && cd repo
```
or copy the files from your laptop (run on the laptop):
```bash
scp -P <PORT> run.py score.py sweep.py check_results.py make_devset.py \
    requirements.txt setup_4090.sh root@<IP>:/root/repo/
```

## 3. Setup  [10-20 min]
```bash
read -s HF_TOKEN && export HF_TOKEN     # paste the token; it is not echoed or saved
bash setup_4090.sh
source venv/bin/activate
```
Check the output: `torch 2.10.0 ... built for CUDA 12.8 | GPU available: True | NVIDIA GeForce RTX 4090`.

## 4. Smoke test  [2 min]
```bash
python run.py --input finevideo/data --output smoke.jsonl --limit 1
head -3 smoke.jsonl
```
Optional: prove it needs no internet (works only if the container allows it):
```bash
unshare -n python run.py --input finevideo/data --output offline.jsonl --limit 1 \
  && echo "OFFLINE OK"
```

## 5. Stride sweep  [about 50 min, time-boxed]
```bash
python sweep.py log --files finevideo/data/train-0000[2-6]-*.parquet \
    --videos 150 --window 90 --minutes 50 --out sweep_log.jsonl
python sweep.py replay --log sweep_log.jsonl | tee sweep_report.txt
```
Read the `verdict on p5` column: for each stride, the scores at the 5th
percentile of 1000 bootstrap resamples over videos ("worst plausible").

## 6. Real 4090 speed per stride  [about 10 min]
Same 6 videos for each:
```bash
for s in 2 3 4; do
  python run.py --input finevideo/data --output t$s.jsonl --limit 6 --stride $s 2>&1 | tail -5
done 2>&1 | tee speed_report.txt
```

## 7. Choose the stride
Suggested rule: the largest stride whose p5 scores keep a comfortable margin
(e.g. >= 0.02) on all three checks. If it is not 3, change `DEFAULT_STRIDE`
in run.py (and the comment above it).

## 8. Full run on all 50 files  [hours]
```bash
ls finevideo/data/*.parquet | wc -l     # must be 50 (background download)
python run.py --input finevideo/data --output results.jsonl --resume 2>&1 | tee fullrun.log
```
`--resume` is harmless on a fresh start. If anything stops the run, run the
same command again: finished videos are kept, the rest is redone.
The last lines of `fullrun.log` give the wall time for HARDWARE.md.

## 9. Check, compress, copy back  [5 min]
```bash
python check_results.py --input finevideo/data --results results.jsonl
gzip -k results.jsonl                  # -> results.jsonl.gz
nproc; lscpu | grep "Model name"; free -h; df -h .; lsblk -d -o NAME,ROTA,SIZE
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```
(`ROTA 0` = SSD.) Copy to your laptop (run on the laptop):
```bash
scp -P <PORT> root@<IP>:/root/repo/{results.jsonl.gz,fullrun.log,sweep_report.txt,speed_report.txt} .
```

## 10. Destroy the instance
Vast keeps charging for storage until the instance is destroyed.
