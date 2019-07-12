import os
import glob
import random
import numpy as np
import aug_util as aug
import wv_util as wv


TRAINING_IMAGE_DIR = os.path.join(os.path.expanduser('~'), 'WORK', 'data', 'xView', 'train_images')
OLD_ROOT = "../xView"
NEW_ROOT = "../xView-meta"
NUM_TEST_IMAGES = 168

# Create list of all image paths in xView training set
tif_path_list = glob.glob(TRAINING_IMAGE_DIR + '/*.tif')
# Create list of all tif names from tif_path_list
tif_list = [x.split('/')[-1] for x in tif_path_list]


### Filter out paths of images that are corrupted
chip_shape = (100,100)

# Load all info from geojson
all_coords, all_chips, all_classes = wv.get_labels(OLD_ROOT + "/xView_train.geojson")

# Iterate over each tif, and chip 
viable_test_tifs = tif_list.copy()
for tif in tif_list:
	# Get the info relevant to this single full frame .tif
	ff_coords = all_coords[all_chips==tif]
	ff_classes = all_classes[all_chips==tif].astype(np.int64)

	# Chip the image into smaller pieces
	arr = wv.get_image(OLD_ROOT + "/train_images/" + tif)  
	c_img, c_box, c_cls = wv.chip_image(img=arr, coords=ff_coords, classes=ff_classes, shape=chip_shape)
	num_chips = len(c_img)

	# Iterate over chips for this tif
	for chip in c_img:
		# If a chip is corrupted, remove this tif from test viable list
		if chip.min() == chip.max():
			print("found corrupted tif:", tif)
			viable_test_tifs.remove(tif)
			break

# We now have an uncorrupted list of tif names to choose our test set from
print("viable test tifs:", len(viable_test_tifs), "/", len(tif_list))

# Randomly choose test tifs
random.shuffle(viable_test_tifs)
test_tifs = viable_test_tifs[:NUM_TEST_IMAGES]

# Store the rest to train tifs list
train_tifs = [x for x in tif_list if x not in test_tifs]

# Remove .tif extension from both lists
train_tifs_no_ext = [x.split('.')[0] for x in train_tifs]
test_tifs_no_ext = [x.split('.')[0] for x in test_tifs]
print("train_tifs:", train_tifs_no_ext)
print("test_tifs:", test_tifs_no_ext)

# Write lists to file
with open(NEW_ROOT + "/ff_train.txt", "w") as f:
	for item in train_tifs_no_ext:
		f.write("%s\n" % item)

with open(NEW_ROOT + "/ff_val.txt", "w") as f:
	for item in test_tifs_no_ext:
		f.write("%s\n" % item)

print("\n\nDone! Files written to: {}".format(NEW_ROOT))

