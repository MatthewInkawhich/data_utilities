import os
import glob
import random
import numpy as np
import aug_util as aug
import wv_util as wv


TRAINING_IMAGE_DIR = os.path.join(os.path.expanduser('~'), 'WORK', 'xview_data', 'xView', 'train_images')
META_ROOT = os.path.join(os.path.expanduser('~'), 'WORK', 'xview_data', 'xView-coco-600', 'meta')
NUM_TEST_IMAGES = 169
#NUM_TEST_IMAGES = 4

# Create list of all image paths in xView training set
tif_path_list = glob.glob(TRAINING_IMAGE_DIR + '/*.tif')

# Create list of all tif names from tif_path_list
tif_list = [x.split('/')[-1] for x in tif_path_list]


# Randomly choose test tifs
random.shuffle(tif_list)
test_tifs = tif_list[:NUM_TEST_IMAGES]

# Store the rest to train tifs list
train_tifs = [x for x in tif_list if x not in test_tifs]

# Create directories if they don't exist
if not os.path.isdir(META_ROOT):
    os.makedirs(META_ROOT)

# Write lists to file
with open(META_ROOT + "/ff_train.txt", "w") as f:
	for item in train_tifs:
		f.write("%s\n" % item)

with open(META_ROOT + "/ff_val.txt", "w") as f:
	for item in test_tifs:
		f.write("%s\n" % item)

print("\n\nDone! Files written to: {}".format(META_ROOT))

