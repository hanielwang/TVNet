#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --job-name=testing
#SBATCH --partition gpu
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --workdir ./

#cd $SLURM_SUBMIT_DIR



#################################################################
module load languages/anaconda2/5.0.1.tensorflow-1.6.0

python TEM_train.py
python TEM_test.py

python VEM_create_windows.py --window_length 15 --window_stride 5
python VEM_create_windows.py --window_length 5 --window_stride 2

python VEM_train.py --voting_type start --window_length 15 --window_stride 5
python VEM_test.py --voting_type start --window_length 15 --window_stride 5

python VEM_train.py --voting_type end --window_length 15 --window_stride 5
python VEM_test.py --voting_type end --window_length 15 --window_stride 5

python VEM_train.py --voting_type start --window_length 5 --window_stride 2
python VEM_test.py --voting_type start --window_length 5 --window_stride 2

python VEM_train.py --voting_type end --window_length 5 --window_stride 2
python VEM_test.py --voting_type end --window_length 5 --window_stride 2

####################################################################
module load languages/anaconda3/2019.07-3.6.5-tflow-1.14
python PEM_train.py
python proposal_generation.py
python post_postprocess.py
