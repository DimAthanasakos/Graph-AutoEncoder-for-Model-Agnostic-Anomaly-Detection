#!/usr/bin/bash

# Load pytorch w/GPU support
module load pytorch/2.1.0-cu12 
module list

# Install additional packages
# The modulefiles automatically set the $PYTHONUSERBASE environment variable for you, 
#   so that you will always have your custom packages every time you load that module.
pip install --user \
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
uproot \
matplotlib==3.6.2 \
mplhep==0.3.26 \
PyYAML==6.0.1 \
torchinfo \
pyjet \
coffea \
qpth


