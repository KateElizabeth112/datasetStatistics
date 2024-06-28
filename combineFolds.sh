#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -N TS_combine_folds_sex

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

#ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'
ROOT_DIR='/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator/'

#python3 combineFolds.py -r $ROOT_DIR -v "Age"

python3 combineFolds.py -r $ROOT_DIR -v "Sex" -d "TS"