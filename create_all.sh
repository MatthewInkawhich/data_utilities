#!/bin/bash

python -u create_val_chip_offsets.py 600 &> logs/600_offsets
python -u create_val_chip_offsets.py 200 &> logs/200_offsets
python -u create_val_chip_offsets.py 400 &> logs/400_offsets

echo "Done!!"
