#!/bin/bash

ROOT_DIR='/Users/katecevora/Documents/PhD/data/TotalSegmentator/'
python3 combineFolds.py -r $ROOT_DIR -v "Age" -d "TS"
python3 combineFolds.py -r $ROOT_DIR -v "Sex" -d "TS"

ROOT_DIR='/Users/katecevora/Documents/PhD/data/AMOS_3D/'
python3 combineFolds.py -r $ROOT_DIR -v "Age" -d "AMOS"
python3 combineFolds.py -r $ROOT_DIR -v "Sex" -d "AMOS"

# Run scripts for compiling and formatting results
python3 runResultsProcessing.py -d "AMOS_3D" -v "Age"
python3 runResultsProcessing.py -d "AMOS_3D" -v "Sex"
python3 runResultsProcessing.py -d "TotalSegmentator" -v "Age"
python3 runResultsProcessing.py -d "TotalSegmentator" -v "Sex"