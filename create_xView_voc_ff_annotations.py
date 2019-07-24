'''
This script creates annotations for the 
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
    #           is length 4 and contains [xMin, yMin, xMax, yMax]

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
    

def clip(old, minimum, maximum):
    if old < minimum:
        return minimum
    elif old > maximum:
        return maximum
    return old



""" Code for xView -> xView-voc format
"""
######### Inputs
OLD_ROOT = "../xView"
NEW_ROOT = "../xView-meta"
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

# Create directories if they don't exist
if not os.path.isdir(NEW_ROOT+"/Annotations"):
    os.makedirs(NEW_ROOT+"/Annotations")


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

    # Read in image to get size
    arr = wv.get_image(OLD_ROOT+"/train_images/"+unique_tif)

    # Get the info relevant to this single full frame .tif
    ff_coords = all_coords[all_chips==unique_tif]
    ff_classes = all_classes[all_chips==unique_tif].astype(np.int64)
    print("\tTotal Num Targets: ",len(ff_classes))

    # Clip boxes between 0 and max
    clipped_boxes = []
    for box in ff_coords:
        xMin,yMin,xMax,yMax = box
        xMin = int(clip(xMin, 0, arr.shape[1]-1))
        yMin = int(clip(yMin, 0, arr.shape[0]-1))
        xMax = int(clip(xMax, 0, arr.shape[1]-1))
        yMax = int(clip(yMax, 0, arr.shape[0]-1))
        clipped_boxes.append([xMin, yMin, xMax, yMax])
                
    # Remove non-recognized class instances
    final_classes = []
    final_boxes = []
    for j, box in enumerate(clipped_boxes):
        if ff_classes[j] in class_names_LUT.keys():
            final_boxes.append(box)
            final_classes.append(ff_classes[j])


    # Construct the saved XML filename
    xml_name = "img_{}.xml".format(unique_tif.split(".")[0])

    # TIME TO WRITE**
    # Convert the integer class labels to english labels
    final_english_classes = [class_names_LUT[lbl] for lbl in final_classes]
    assert(len(final_boxes) == len(final_english_classes))

    # Create and populate the XML file for this chip
    XMLWriter_VOCformat(NEW_ROOT+"/Annotations/"+xml_name, "xView-meta", unique_tif, arr.shape[1], arr.shape[0], final_english_classes, final_boxes)

