# Hardware used for the full run

Rented on Vast.ai (on-demand), Ubuntu container with the PyTorch (Vast) image.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB (23028 MiB reported) |
| Driver | 620.70.06 |
| CPU | AMD EPYC 7763 (64 cores / 128 threads on the host); about 18 vCPUs allocated to the instance (per the Vast listing, not measured) |
| RAM | Shared host with 251 GiB total; the instance's own allocation was not measured |
| Disk | NVMe SSD (ROTA 0), 100 GB container disk |
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
| Checks | `check_results.py`: OK (no missing videos, no gaps or duplicate seconds, 0 errors) |

## Stride choice

Stride sweep on 150 videos (random 90 s windows from files 2-6): every stride from 1 to 10
passed at the 5th percentile of 1000 bootstrap resamples. Stride 4 margin was +0.040.
Speed on the same 6 videos (stride 2 / 3 / 4 / 6 / 8): 585 / 739 / 863 / 858 / 1174 video
frames/s. See `sweep_report.txt` and `speed_report.txt`.
