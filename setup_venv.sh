module load cuda/8.0.44
module load cudnn/7.0
module load qt
module load python/3.6

virtualenv ~/projects/def-dprecup/zafarali/venvs/RVI
source ~/projects/def-dprecup/zafarali/venvs/RVI

pip install numpy scipy --no-index
pip install matplotlib seaborn pandas --no-index
pip install tensorboardX
pip install gym==0.10.3
pip install /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/avx2/torch_cpu-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install -e git+https://github.com/zafarali/mlresearchkit.git#egg=mlresearchkit
pip install -e git+https://github.com/zafarali/policy-gradient-methods.git#egg=pg_methods
pip install test_tube
pip install pytest==3.0.7

python3 -m pytest ./tests/