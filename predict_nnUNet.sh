#!/bin/bash
#PBS -l walltime=30:00:00
#PBS -l select=1:ncpus=15:mem=120gb:ngpus=1:gpu_type=RTX6000
#PBS -N predict_TS_age

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

# Set environment variables
#ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'
ROOT_DIR='/rds/general/user/kc2322/projects/cevora_phd/live/TotalSegmentator/'

DATASET="TotalSegmentator"
experiments=("Dataset500_Age0" "Dataset501_Age0" "Dataset502_Age0"
             "Dataset600_Age1" "Dataset601_Age1" "Dataset602_Age1"
             "Dataset700_Age2" "Dataset701_Age2" "Dataset702_Age2"
             "Dataset800_Age3" "Dataset801_Age3" "Dataset802_Age3"
             "Dataset900_Age4" "Dataset901_Age4" "Dataset902_Age4")

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

for number in {0..14}; do
    EXPERIMENT=${experiments[number]}
    TASK=${EXPERIMENT:7:3}

    # Inference
    INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/"$DATASET"/imagesTs"
    OUTPUT_FOLDER=$ROOT_DIR"inference/"$DATASET"/all"

    echo $TASK
    echo $DATASET
    echo $EXPERIMENT
    echo $INPUT_FOLDER
    echo $OUTPUT_FOLDER

    #nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $TASK -c 3d_fullres -f all -chk checkpoint_best.pth

    # Run python script to evaluate results
    python3 processResults.py -d $DATASET -e $EXPERIMENT -r $ROOT_DIR

done

python3 combineFolds.py -r $ROOT_DIR -v "Age"
