#!/bin/bash
#SBATCH --output=//global/homes/d/dimathan/gae_for_anomaly/Results/unsupervised/qqq/s_b_p03/0929_n30_un234_%j.txt
#SBATCH -A alice_g
#SBATCH -C gpu
#SBATCH -q regular 
#SBATCH -t 3:30:00
#SBATCH -n 4

cd gae_for_anomaly/
source ./init_perlmutter.sh 

python -u analysis/steer_analysis.py -c config/config1.yaml -regen --n_runs 4 -s 1
python -u analysis/steer_analysis.py -c config/config2.yaml -regen --n_runs 4 -s 1
python -u analysis/steer_analysis.py -c config/config3.yaml -regen --n_runs 4 -s 1