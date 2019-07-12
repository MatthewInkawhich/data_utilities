# NAI

'''
This script is a utility to convert the original xView dataset to a VOC2007
  like layout. It will create a second xView dataset directory where it 
  will store the extracted image chips and annotations.
'''
from __future__ import print_function
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import skimage.io 
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom

''' Code for XMLWriter for VOC format
'''
def prettify(elem):
	"""Return a pretty-printed XML string for the Element.
	"""
	rough_string = ElementTree.tostring(elem, 'utf-8')
	reparsed = minidom.parseString(rough_string)
	return reparsed.toprettyxml(indent="  ")

def XMLWriter_VOCformat(xml_out_file, data_folder, img_name, img_width, img_height, class_names, boxes):

	# xml_out_file = (string) filename for output xml file 
	# folder = (string) name of dataset to go in 'folder' field
	# img_name = (string) name of corresponding image that this XML annotates
	# width = (int) width of image
	# height = (int) height of image
	# class_names = ([string]) list of human-readable class names. The length equals how many objects are in this image
	# boxes = ([[int,int,int,int], ...] list of lists of integers. List is index-matched with class_names. Each sub-list 
	#			is length 4 and contains [xMin, yMin, xMax, yMax]

	# Write static boiler plate stuff
	#    folder, filename, owner, size, etc...
	annotation = Element('annotation')
	
	folder = SubElement(annotation, 'folder')
	folder.text = data_folder	
	filename = SubElement(annotation, 'filename')
	filename.text = img_name	
	owner = SubElement(annotation, 'owner')
	owner.text = 'NAI'
	segmented = SubElement(annotation, 'segmented')
	segmented.text = '0'

	# Source Element
	source = SubElement(annotation, 'source')
	database = SubElement(source, 'database')
	database.text = "xView-voc Database"
	source_annotation = SubElement(source, 'annotation')
	source_annotation.text = 'xView 2018'
	image = SubElement(source, 'image')
	image.text = 'xView'

	# Size Element
	size = SubElement(annotation, 'size')
	width = SubElement(size, 'width')
	width.text = str(img_width)
	height = SubElement(size, 'height')
	height.text = str(img_height)
	depth = SubElement(size, 'depth')
	depth.text = "3"

	# Object Elements		
	for i in range(len(class_names)):
		obj = SubElement(annotation, 'object')
		name = SubElement(obj, 'name')
		name.text = class_names[i]
		pose = SubElement(obj, 'pose')
		pose.text = 'top'	
		truncated = SubElement(obj, 'truncated')
		truncated.text = '0'
		difficult = SubElement(obj, 'difficult')
		difficult.text = '0'
		bndbox = SubElement(obj, 'bndbox')
		xmin = SubElement(bndbox, 'xmin')
		xmin.text = str(boxes[i][0])
		ymin = SubElement(bndbox, 'ymin')
		ymin.text = str(boxes[i][1])
		xmax = SubElement(bndbox, 'xmax')
		xmax.text = str(boxes[i][2])
		ymax = SubElement(bndbox, 'ymax')
		ymax.text = str(boxes[i][3])

	# Get rid of xml version 1.0 heading	
	prettyxml = prettify(annotation)[23:]

	# Save xml here
	f = open(xml_out_file,"w")
	f.write(prettyxml)
	f.close()
	

""" Code for xView -> xView-voc format
"""
######### Inputs
OLD_ROOT = "../xView"
NEW_ROOT = "/bigdata/NFTI/datasets/xView2018/xView-voc"
#NEW_ROOT = "../xView-voc"
CHIP_SHAPE = [500,600]
#CHIP_SHAPE = [400]
BOX_AREA_THRESH = 10
#########

# Create the Class # -> Class Label Dictionary
class_names_LUT = {}
with open('xview_class_labels_formatted.txt') as f:
	for row in csv.reader(f):
		class_names_LUT[int(row[0].split(":")[0])] = row[0].split(":")[1]

# Load all of the labels from .geojson
all_coords, all_chips, all_classes = wv.get_labels(OLD_ROOT + "/xView_train.geojson")

print("Full Dataset Stats:")
print(all_coords.shape)
print(all_chips.shape)
print(all_classes.shape)

'''
# Find area of smallest target
# Smallest area = 0
# Largest area = 10332468
smallest_area = 100000
largest_area = 0
for bbox in all_coords:
	xmin,ymin,xmax,ymax = bbox
	area = (xmax-xmin)*(ymax-ymin)
	if area < smallest_area:
		smallest_area = area
	if area > largest_area:
		largest_area = area
print("Smallest Area: ",smallest_area)
print("Largest Area: ",largest_area)
exit()
'''

# Get all of the unique .tif names from all_chips
tif_names = np.unique(all_chips)

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
	#print("\tTotal Num Targets: ",len(ff_classes))

	# Add multiscale loop
	for scale in CHIP_SHAPE:

		# Chip the image into smaller pieces
		arr = wv.get_image(OLD_ROOT+"/train_images/"+unique_tif)  
		c_img, c_box, c_cls = wv.chip_image(img=arr, coords=ff_coords, classes=ff_classes, shape=(scale,scale))
		num_chips = len(c_img)
		#print("\tNum Chips: ",num_chips)

		# For each image chip (i in range(num_chips))
		for i in range(num_chips):	

			#print("\t\tChip #: ",i)

			# Calculate the center of the chip
			center = (int(c_img[i].shape[0]/2),int(c_img[i].shape[1]/2))

			# For each of the desired rotation degrees
			for val in range(0,360,10):
				
				# Construct the saved chip name
				chip_name = "img_{}_{}_{}x{}_rot{}.png".format(unique_tif.split(".")[0], i, scale, scale, val)
			
				# if the file already exists, skip over it
				if os.path.isfile(NEW_ROOT+"/JPEGImages/"+chip_name):
					continue	
	
				# Rotate the original chip and get the updated image/boxes/classes		
				tmp_img,tmp_box,tmp_cls = aug.rotate_image_and_boxes(c_img[i], val, center, c_box[i], c_cls[i])

				# Git rid of very small boxes that are artifacts of chipping
				final_boxes = []
				final_classes = []
					
				# Eliminate clipped boxes whose area is too small	
				for j,box in enumerate(tmp_box):
					xMin,yMin,xMax,yMax = box
					box_area = (xMax-xMin)*(yMax-yMin)
					if box_area > BOX_AREA_THRESH:
						# Excludes odd error cases
						if tmp_cls[j] in class_names_LUT.keys():		
							#final_boxes.append(box.tolist())
							final_boxes.append(box)
							final_classes.append(tmp_cls[j])

				# Enforce rule that if the rot0 chip has no boxes, then none of the other rotations will have boxes so skip the chip all together	
				if (val == 0) and (len(final_classes)==0):
					#print("rot0 has no targets, skipping whole chip...")
					break
			
				# Get rid of images with no targets and all black images. Only save the images that
				#  contain targets and have visible features
				if not((final_classes == []) and (final_boxes == [])) and (tmp_img.max() != tmp_img.min()):
						
					# Construct the saved chip name
					#chip_name = "img_{}_{}_{}x{}_rot{}.png".format(unique_tif.split(".")[0], i, scale, scale, val)

					# Construct the saved XML filename
					xml_name = chip_name.replace(".png",".xml")

					# Convert the integer class labels to english labels
					final_english_classes = [class_names_LUT[lbl] for lbl in final_classes]
					assert(len(final_boxes) == len(final_english_classes))

					# Create and populate the XML file for this chip
					XMLWriter_VOCformat(NEW_ROOT+"/Annotations/"+xml_name, "xView-voc", chip_name, scale, scale, final_english_classes, final_boxes)

					# Save the chipped image 
					skimage.io.imsave( NEW_ROOT+"/JPEGImages/"+chip_name, tmp_img, quality=100)	
					#cv2.imwrite( NEW_ROOT+"/JPEGImages/"+chip_name, c_img[i], cv2.CV_IMWRITE_JPEG_QUALITY,100)	


