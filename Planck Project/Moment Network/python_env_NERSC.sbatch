#!/bin/bash

#SBATCH -A mp107d_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=py_env
#SBATCH -o python_env.log
#SBATCH -e python_env.err
#SBATCH --export=NONE

set -eo pipefail

set +u
module purge
set -u

echo "host=$(hostname)"
echo "date=$(date)"
echo

echo "=== System python3 ==="
command -v python3 || true
python3 --version || true
python3 - <<'PY' || true
import sys
print("sys.executable=", sys.executable)
print("version=", sys.version)
PY
echo

echo "=== NERSC pytorch module ==="
set +u
module load pytorch
set -u
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
command -v python || true
python --version || true
python - <<'PY' || true
import sys
print("sys.executable=", sys.executable)
print("version=", sys.version)
import numpy
print("numpy=", numpy.__version__)
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
PY
echo

echo "=== Hard-coded cmbenv python3, no modules ==="
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
/pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/python3 --version || true
/pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/python3 - <<'PY' || true
import sys
print("sys.executable=", sys.executable)
print("version=", sys.version)
import encodings
print("encodings ok")
PY
echo

echo "=== cmbenv module, no conda activate ==="
set +u
module use /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/modulefiles
module load cmbenv
set -u
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
command -v python || true
python --version || true
command -v python3 || true
python3 --version || true
python3 - <<'PY' || true
import sys
print("sys.executable=", sys.executable)
print("version=", sys.version)
import numpy
print("numpy=", numpy.__version__)
try:
    import torch
    print("torch=", torch.__version__)
except Exception as exc:
    print("torch import failed:", repr(exc))
PY
echo

echo "=== cmbenv conda activate ==="
set +u
source /pscratch/sd/a/atsouros/cmbenv/master-0.0.1/conda/bin/activate || true
set -u
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
command -v python || true
python --version || true
command -v python3 || true
python3 --version || true
python3 - <<'PY' || true
import sys
print("sys.executable=", sys.executable)
print("version=", sys.version)
import numpy
print("numpy=", numpy.__version__)
try:
    import torch
    print("torch=", torch.__version__)
except Exception as exc:
    print("torch import failed:", repr(exc))
PY

echo
echo "done=$(date)"
