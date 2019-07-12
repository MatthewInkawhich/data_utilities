# This script counts the max number of GT boxes in any chip

import os
import glob
import random
import numpy as np
import aug_util as aug
import wv_util as wv


TRAINING_IMAGE_DIR = os.path.join(os.path.expanduser('~'), 'WORK', 'data', 'xView', 'train_images')
OLD_ROOT = "../xView"

# Create list of all image paths in xView training set
tif_path_list = glob.glob(TRAINING_IMAGE_DIR + '/*.tif')
# Create list of all tif names from tif_path_list
tif_list = [x.split('/')[-1] for x in tif_path_list]

# Chip shape to consider
chip_shape = (700,700)

# Load all info from geojson
all_coords, all_chips, all_classes = wv.get_labels(OLD_ROOT + "/xView_train.geojson")


max_gt_boxes = 0

# Iterate over each tif, and chip 
for e, tif in enumerate(tif_list):
	print("{} / {}".format(e, len(tif_list)))
	# Get the info relevant to this single full frame .tif
	ff_coords = all_coords[all_chips==tif]
	ff_classes = all_classes[all_chips==tif].astype(np.int64)

	# Chip the image into smaller pieces
	arr = wv.get_image(OLD_ROOT + "/train_images/" + tif)  
	c_img, c_box, c_cls = wv.chip_image(img=arr, coords=ff_coords, classes=ff_classes, shape=chip_shape)
	num_chips = len(c_img)

	# Iterate over chips for this tif
	for i in range(num_chips):
		# If chip has no gt boxes, skip
		if len(c_box[i]) == 1 and c_box[i].all() == 0:
			continue
		if len(c_box[i]) > max_gt_boxes:
			max_gt_boxes = len(c_box[i])
			max_gt_chip_num = i
			max_gt_tif = tif
			print("new best:", max_gt_boxes, max_gt_tif, max_gt_chip_num)

print("max_gt_boxes:", max_gt_boxes)
print("in tif:", max_gt_tif)
print("chip num:", max_gt_chip_num)
