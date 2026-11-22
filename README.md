# Video indexer

`run.py` reads a folder of FineVideo Parquet files and writes one JSON line per
second of every video: the counts of person, car, dog, laptop and cell phone
detected by YOLO11l.

Tested on Ubuntu 24.04, Python 3.11, NVIDIA RTX 4090 (driver with CUDA 12 support).

## Install

System packages (needed by OpenCV):
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

Python 3.11 virtual environment. Ubuntu 24.04 has no python3.11 package, so use `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --seed -p 3.11 venv
source venv/bin/activate
```
(If `python3.11` is already installed, `python3.11 -m venv venv` works too.)

Python packages, with exact versions:
```bash
pip install -r requirements.txt
```

## Preparation command

Download the model weights once (the run itself does not need internet after this):
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

The defaults are the submitted configuration: every 4th frame of each second,
FP16 inference, batch size 32. For the exact baseline (every frame, FP32) use
`--stride 1 --no-half`.

Useful options: `--limit N` (first N videos only), `--resume` (keep finished
videos and continue after an interruption).

Each output line looks like:
```json
{"video": "train-00000-of-01357.parquet:0", "second": 0, "objects": [], "people": 0}
```

Compress the results with `gzip -k results.jsonl` to get `results.jsonl.gz`.
