#!/usr/bin/env bash
# One-time setup on the rented GPU machine (Ubuntu, NVIDIA driver with CUDA 12+).
#
#   export HF_TOKEN=hf_xxx          # your read token; never write it into a file
#   bash setup_4090.sh
#
# Creates a Python 3.11 venv in ./venv (like the graders do), installs
# requirements.txt, downloads yolo11l.pt, then downloads the dataset:
# files 2-6 first (needed for the stride sweep), then all 50 in the background.
set -euo pipefail
cd "$(dirname "$0")"

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "=== system packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq libgl1 libglib2.0-0 tmux curl git >/dev/null

echo "=== Python 3.11 virtual environment"
if ! command -v python3.11 >/dev/null; then
    $SUDO apt-get install -y -qq python3.11 python3.11-venv >/dev/null || true
fi
if command -v python3.11 >/dev/null && python3.11 -m venv venv 2>/dev/null; then
    echo "using system python3.11"
else
    echo "no usable python3.11 from apt, using uv to get one"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --seed -p 3.11 venv
fi
source venv/bin/activate
python --version
pip install -q -U pip
pip install -q -r requirements.txt
pip install -q huggingface_hub          # only for downloading the dataset

echo "=== GPU check"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda,
      "| GPU available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "CUDA not available - stop and check the driver"
EOF

echo "=== model weights (the brief's preparation step)"
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')" && ls -lh yolo11l.pt

echo "=== dataset: files 2-6 first (for the sweep)"
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
hf download HuggingFaceFV/finevideo --repo-type dataset \
    --include "data/train-0000[2-6]-*" --local-dir finevideo
ls -lh finevideo/data

echo "=== dataset: all 50 files, in the background (log: download.log)"
nohup hf download HuggingFaceFV/finevideo --repo-type dataset \
    --include "data/train-000[0-4]*" --local-dir finevideo > download.log 2>&1 &
echo "background download started (pid $!). Check: ls finevideo/data | wc -l   (expect 50)"
echo "=== setup done. Next: source venv/bin/activate"
