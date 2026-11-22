# Hardware used for the full run

Rented on Vast.ai (on-demand), Ubuntu container with the PyTorch (Vast) image.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (23028 MiB reported) |
| Driver | 620.70.06 |
| CPU | AMD EPYC 7763 (64 cores / 128 threads on the host); about 18 vCPUs allocated to the instance (per the Vast listing, not measured) |
| RAM | Shared host with 251 GiB total; the instance's own allocation was not measured |
| Disk | NVMe SSD (ROTA 0), 100 GB container disk |
| Operating system | Ubuntu 24.04 (inferred from the instance's OpenSSH 9.6p1 Ubuntu-3ubuntu13 banner; not checked with `lsb_release`) |
| Software | Python 3.11.17, torch 2.10.0+cu128 (CUDA 12.8), yolo11l.pt |

## Run settings

`python run.py --input finevideo/data --output results.jsonl --resume`

Defaults: stride 4, FP16, batch 32.

## Full run

| Item | Value |
|---|---|
| Videos | 1650 (50 parquet files) |
| Seconds of video | 461,757 (128.3 hours) |
| Wall time | 10,133.6 s (about 2 h 49 min) |
| Frames decoded | 12,981,615 |
| Frames sent to YOLO | 3,463,822 |
| Time in YOLO | 7,744.6 s (76%) |
| Decode, convert, other | 2,389.1 s |
| Throughput | 1,281 video frames/s |