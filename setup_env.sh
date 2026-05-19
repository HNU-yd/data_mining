#!/usr/bin/env bash
set -euo pipefail

conda create -n data_mining python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate data_mining

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install facenet-pytorch==2.6.0 --no-deps

python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda smoke test", float(y[0, 0]))
PY
