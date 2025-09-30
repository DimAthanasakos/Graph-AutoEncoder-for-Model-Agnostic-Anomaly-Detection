#!/usr/bin/bash

# Load pytorch w/GPU support
module load pytorch/2.6.0
module list

# Install additional packages
# The modulefiles automatically set the $PYTHONUSERBASE environment variable for you, 
#   so that you will always have your custom packages every time you load that module.
python -m pip install --user \
seaborn==0.11.2 \
silx==1.1.2 \
numba==0.57.0 \
numpy==1.24.4 \
cython==0.29.30 \
blosc2==2.0.0 \
triton \
energyflow \
vector \
awkward \
uproot 

module use "$HOME/heppy/modules"
module load heppy 

echo "Modules loaded"

export PYTHONPATH="/global/common/software/nersc9/pytorch/2.6.0/lib/python3.12:$PYTHONPATH"
cd "$HOME/gae_for_anomaly"
alias python='python3.12'
echo "Alias set"

# The following packages are already installed by the pytorch module
# matplotlib==3.5.1 \
# networkx==2.7.1 \
# numpy==1.21.2 \
# pandas==1.4.1 \
# pyyaml==6.0 \
# scikit-learn==1.0.2 \
# torch==1.11 \
# torch-geometric==2.0.4 \
# torch-scatter==2.0.9 \
# torch-sparse==0.6.13
