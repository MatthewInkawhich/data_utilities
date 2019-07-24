'''
This script 
'''
from __future__ import print_function
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os
import skimage.io 
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom

""" Code for xView -> xView-voc format
"""
### Read input arg
parser = argparse.ArgumentParser(description='Create Annotations and images for xview chips')
parser.add_argument('chip_size', type=int)
args = parser.parse_args()

######### Inputs
CHIP_SIZE = args.chip_size
OLD_ROOT = "../xView"
NEW_ROOT = "../xView-voc-{}".format(CHIP_SIZE)
IMAGESETS_PATH = os.path.join(NEW_ROOT, "ImageSets", "Main")
CHIP_SHAPE = (CHIP_SIZE, CHIP_SIZE)
OVERLAP = 0.20
#########

# Load all of the labels from .geojson
all_coords, all_chips, all_classes = wv.get_labels(OLD_ROOT + "/xView_train.geojson")

print("Full Dataset Stats:")
print(all_coords.shape)
print(all_chips.shape)
print(all_classes.shape)


# Get all of the unique .tif names from all_chips
tif_names = np.unique(all_chips)

rows = []

xyz = 0
# For each unique .tif
for unique_tif in tif_names:

    xyz += 1
    print("Working on: [{} / {}] {} ".format(xyz, len(tif_names), unique_tif))

    # Make sure the file exists
    if not os.path.isfile(OLD_ROOT+"/train_images/"+unique_tif):
        continue

    # Get the info relevant to this single full frame .tif
    ff_coords = all_coords[all_chips==unique_tif]
    ff_classes = all_classes[all_chips==unique_tif].astype(np.int64)
    print("\tTotal Num Targets: ",len(ff_classes))

    # Chip the image into smaller pieces
    arr = wv.get_image(OLD_ROOT+"/train_images/"+unique_tif)  
    #c_img, c_box, c_cls = wv.chip_image(img=arr, coords=ff_coords, classes=ff_classes, shape=CHIP_SHAPE)
    c_img, c_box, c_cls, c_offsets = wv.chip_image_overlap(img=arr, coords=ff_coords, classes=ff_classes, shape=CHIP_SHAPE, overlap=OVERLAP)
    num_chips = len(c_img)
    print("\tNum Chips: ",num_chips)

    # For each image chip (i in range(num_chips))
    for i in range(num_chips):  

        print("\t\tChip #: ",i)

        # Write offsets to offsets file
        row = [unique_tif.split('.')[0], i, c_offsets[i][0], c_offsets[i][1]]
        rows.append(row)


print("done chipping, time to write csv!")
with open(IMAGESETS_PATH+"/chip_offsets.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(rows)
print("done writing csv!")
        
