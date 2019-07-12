#!/bin/bash

python -u create_xView_voc_dataset_with_rotations.py 200 &> logs/v200
python -u create_xView_voc_dataset_with_rotations.py 400 &> logs/v400
python -u create_xView_voc_dataset_with_rotations.py 600 &> logs/v600
python -u create_xView_voc_dataset_with_rotations.py 800 &> logs/v800

echo "Done!!"
