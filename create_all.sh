#!/bin/bash

python -u create_xView_voc_dataset_with_rotations.py 200 &> logs/v200_newval
python -u create_xView_voc_dataset_with_rotations.py 400 &> logs/v400_newval
python -u create_xView_voc_dataset_with_rotations.py 600 &> logs/v600_newval
python -u create_xView_voc_dataset_with_rotations.py 800 &> logs/v800_newval

echo "Done!!"
