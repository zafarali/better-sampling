#!/usr/bin/env bash
# How to run: PATH_TO_VENV="/path/to/venv" bash setup_venv.sh
module load cuda/8.0.44
module load cudnn/7.0
module load qt
module load python/3.6

[ -z "$PATH_TO_VENV" ] && echo "Please provide PATH_TO_VENV" && exit 1;


virtualenv $PATH_TO_VENV
source $PATH_TO_VENV/bin/activate

pip install numpy scipy --no-index
pip install matplotlib seaborn pandas --no-index
pip install tensorboardX
pip install gym==0.10.3
pip install /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/avx2/torch_cpu-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install -e git+https://github.com/zafarali/mlresearchkit.git#egg=mlresearchkit
pip install -e git+https://github.com/zafarali/policy-gradient-methods.git#egg=pg_methods
pip install test_tube
pip install pytest==3.0.7
pip install -e .

python3 -m pytest ./tests/

echo "Virtual environment has been setup at:"
echo $PATH_TO_VENV