# NAI

'''
This script is a utility to convert the original xView dataset to a VOC2007
  like layout. It will create a second xView dataset directory where it 
  will store the extracted image chips and annotations.
'''

import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv

chip_name = "1052.tif"

arr = wv.get_image("../xView/train_images/"+chip_name)

#### Load labels for whole dataset
# - coords.shape = (601937, 4)
# - chips.shape = (601937,)
# - classes.shape = (601937,)
# This makes three flat, index aligned lists, where all of the chips from all images are
#   assembled. If we want all of the info for one chip, we look up the chip name in chips
#   list and use those indexes to access the other lists.
coords, chips, classes = wv.get_labels("../xView/xView_train.geojson")

#print coords[0] # = [2712, 1145, 2746, 1177]
#print chips[0] # = 2355.tif
#print classes[0] # = 73.0

# Get info specific to our chip
coords = coords[chips==chip_name]
classes = classes[chips==chip_name].astype(np.int64)

print("# Objects In Image: ",len(classes))

# Create a Class # -> Class Label Map
labels = {}
with open('xview_class_labels.txt') as f:
	for row in csv.reader(f):
		labels[int(row[0].split(":")[0])] = row[0].split(":")[1]


# Print the class names that are in this image
#print("Classes In this Image: ",[labels[i] for i in np.unique(classes)])

#### Chip the image into smaller sized chips
"""
  Chip an image and get relative coordinates and classes.  Bounding boxes that pass into
     multiple chips are clipped: each portion that is in a chip is labeled. For example,
     half a building will be labeled if it is cut off in a chip. If there are no boxes,
     the boxes array will be [[0,0,0,0]] and classes [0].
 Note: This chip_image method is only tested on xView data-- there are some image manipulations that can mess up different images.

  Args:
     img: the image to be chipped in array format
     coords: an (N,4) array of bounding box coordinates for that image
     classes: an (N,1) array of classes for each bounding box
     shape: an (W,H) tuple indicating width and height of chips

  Output:
     An image array of shape (M,W,H,C), where M is the number of chips,
     W and H are the dimensions of the image, and C is the number of color
     channels.  Also returns boxes and classes dictionaries for each corresponding chip.
"""
# c_img.shape = (M,W,H,C) = (6,1000,1000,3)
# c_box = dictionary keyed by integer and values is array of arrays. Each individual array
#         is length 4 and is the relative bbox coords for an object. Each individual array
#         is organized as [xmin, ymin, xmax, ymax]
#   i.e. c_box = {
#					0: array( [[100, 150, 210, 290],
#								[130, 290, 540, 570],
#								...
#							])
#					1: array([ [...], [...], ... ])
#					...
#				}
#
# c_cls = dictionary keyed by integer and values are arrays of ints which are class labels.			
#   i.e. c_cls = {
#					0: [10, 15, 21, 29, 71, 20]
#					1: [30, 55, 22, 19, 61, 30]
#					...
#				}
# The keys of the dictionaries match up with len(c_img). The details for chip
#   c_img[0] are stored in c_box[0] and c_cls[0]. So, len(c_img) == len(c_box) == len(c_cls)
c_img, c_box, c_cls = wv.chip_image(img=arr, coords=coords, classes=classes, shape=(700,700))

# Augment the data
ind = np.random.choice(range(c_img.shape[0]))
center = (int(c_img[ind].shape[0]/2),int(c_img[ind].shape[1]/2))

for deg in range(0, 360, 45):
	rot_im, rot_boxes = aug.rotate_image_and_boxes(c_img[ind], deg, center, c_box[ind])
	pim = aug.draw_bboxes(rot_im, rot_boxes)
	plt.imshow(pim)
	plt.title("deg: {}".format(deg))
	plt.show()


