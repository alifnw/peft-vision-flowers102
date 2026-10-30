#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.runtime/uv-cache"
export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p .runtime manifests logs
uv_bin="${PEFT_UV:-$(command -v uv || true)}"
training_python="${PEFT_TRAINING_PYTHON:-python3.10}"
if [ -z "$uv_bin" ]; then
    printf '%s\n' 'uv is required on PATH, or set PEFT_UV to its executable.' >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    "$uv_bin" venv --python "$training_python" .venv
fi

.venv/bin/python - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit('The recorded GPU environment requires Python 3.10. Set PEFT_TRAINING_PYTHON when creating it.')
PY

"$uv_bin" pip install --python .venv/bin/python \
    'https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-linux_x86_64.whl' \
    'https://download.pytorch.org/whl/cu117/torchvision-0.14.0%2Bcu117-cp310-cp310-linux_x86_64.whl' \
    'numpy==1.24.4' 'Pillow==9.5.0'
"$uv_bin" pip install --python .venv/bin/python -r requirements-training.txt
"$uv_bin" pip check --python .venv/bin/python
"$uv_bin" --color never pip freeze --python .venv/bin/python > manifests/requirements-training-resolved.txt

jupyter_data_home="${JUPYTER_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/jupyter}"
kernel_path="$jupyter_data_home/kernels/islab-peft-vision"
if [ ! -e "$kernel_path" ]; then
    .venv/bin/python -m ipykernel install --user --name islab-peft-vision \
        --display-name 'Python (ISLab PEFT)'
else
    printf '%s\n' 'Kernel already exists. Its configuration was not overwritten.'
fi

.venv/bin/python scripts/prepare_weights.py
printf '%s\n' 'SETUP_COMPLETE'
