# This script is to be called after creating the JPEGImages and Annotations.
#  Here, we create the ImageSets/Main/train.txt val.txt. Since xView only has
#  labels for the training set, we will split the train set into a train and
#  val set. Here we randomly sample from the full train set to create them.

import os
import glob
import random
import sys

# Directory to write train.txt and val.txt files to
CHIP_SIZE = 800
IMAGESETS_DIR = "/raid/inkawhmj/WORK/data/xView-voc-{}/ImageSets/Main".format(CHIP_SIZE)
ANNOTATIONS_DIR = "/raid/inkawhmj/WORK/data/xView-voc-{}/Annotations".format(CHIP_SIZE)
IMAGES_DIR = "/raid/inkawhmj/WORK/data/xView-voc-{}/JPEGImages".format(CHIP_SIZE)
META_DIR = "/raid/inkawhmj/WORK/data/xView-meta"
FF_TRAIN = META_DIR + "/ff_train.txt"
FF_VAL = META_DIR + "/ff_val.txt"

# read in all filenames from annotations dir into list
print("Reading in all filenames...")
files = glob.glob(ANNOTATIONS_DIR+"/*.xml")
num_files = len(files)
print("Read in {} files".format(num_files))
files = [f.split("/")[-1][:-4] for f in files]

# Define lambda function to extract ffnum from filename
get_ff_id = lambda f : f.split('_')[1]
get_ff_rot = lambda f : f.split('_')[-1]


### Write train.txt and val.txt
# First, read ff image IDs from ff_train.txt and ff_val.txt into a list
train_ids = [line.rstrip('\n') for line in open(FF_TRAIN)]
val_ids = [line.rstrip('\n') for line in open(FF_VAL)]
# Open file handlers
train_fh = open(IMAGESETS_DIR+"/train.txt", "w")
val_fh = open(IMAGESETS_DIR+"/val.txt", "w")

# Loop thru all chip files in Annotations dir
print("Writing files...")
for fname in files:
	fid = get_ff_id(fname)
	# If ID of current image chip matches a training ID, write this filename to file
	if fid in train_ids:
		train_fh.write(fname+"\n")
	# Else, the ID of current image chip is a val ID, so write this filename to file if it is a rot0 chip
	else:
		if get_ff_rot(fname) == "rot0":
			val_fh.write(fname+"\n")


train_fh.close()
val_fh.close()


print("Done!")

