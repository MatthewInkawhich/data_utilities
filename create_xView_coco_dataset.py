# MatthewInkawhich

'''
This script is a utility to convert the original xView dataset to COCO format.
'''
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import argparse
import os
import json
import skimage.io 
    

class xView_JSON_Dict():
    def __init__(self, class_names_LUT):
        # Initialize ID runners
        self.img_id = 0
        self.ann_id = 0
        # Initialize head
        self.head = {}
        # Add info branch
        self.head["info"] = {
                "description": "xView Dataset for 2018 Challenge",
                "url": "http://xviewdataset.org",
                "version": "1.0",
                "year": 2018,
                "contributor": "DIUx",
                "date_created": "2018/01/01"
        }
        # Add licenses branch
        self.head["licenses"] = [{
                "url": "http://xviewdataset.org",
                "id": 1,
                "name": "xView License"
        }]
        # Initialize images branch
        self.head["images"] = []
        # Initialize annotations branch
        self.head["annotations"] = []
        # Add categories branch
        self.head["categories"] = []
        for key, value in class_names_LUT.items():
            self.head["categories"].append({
                "supercategory": "object",
                "id": key,
                "name": value
            })


    # Add an image AND its corresponding annotations to the dict
    def add_image(self, file_name, width, height, boxes, int_classes):
        # Add entry to images branch
        self.head["images"].append({
            "id": self.img_id,
            "width": width,
            "height": height,
            "file_name": file_name,
            "license": 1,
            "flickr_url": "N/A",
            "coco_url": "N/A",
            "date_captured": "2018/01/01"
        })

        # Add entries to annotations branch
        for i in range(len(int_classes)):
            self.head["annotations"].append({
                "id": self.ann_id,
                "image_id": self.img_id,
                "category_id": int_classes[i],
                "segmentation": [],
                "area": boxes[i][2] * boxes[i][3],  # Area is box w * h
                "iscrowd": 0,
                "bbox": boxes[i]
            })
            # Increment annotation id
            self.ann_id += 1

        # Increment image id
        self.img_id += 1

    # Write the dict structure to file
    def write_file(self, out_path, indent=None):
        with open(out_path, 'w') as outfile:
            json.dump(self.head, outfile, indent=indent)


# Define a function for translating full labels to simple labels
def convert_full_to_simple(full_boxes, full_classes):
    full_to_simple_translator = {
        (11,12,13): 1,
        (17,18,19): 2,
        (71,72,73,74,76,77): 3,
        (20,21,23,24,25,27,28,29): 4,
        (33,34,35,36,37,38): 5,
        (40,41,42,44,45,47,49,50,51,52): 6,
        (53,54,55,56,57,59,60,61,62,63,64,65,66,32): 7
    }
    simple_boxes = []
    simple_classes = []
    for j in range(len(full_classes)):
        for k, v in full_to_simple_translator.items():
            if full_classes[j] in k:
                simple_boxes.append(full_boxes[j])
                simple_classes.append(v)
                break
    return simple_boxes, simple_classes



# Define how to clip a value
def clip(old, minimum, maximum):
    if old < minimum:
        return minimum
    elif old > maximum:
        return maximum
    return old
        

######### Inputs
TRAIN = True
CHIP_SIZE = 600
OLD_ROOT = "../xView"
NEW_ROOT = "../xView-coco-{}".format(CHIP_SIZE)
CHIP_SHAPE = (CHIP_SIZE, CHIP_SIZE)
BOX_AREA_THRESH = 20
OVERLAP = 0.20
#########

# Set image path and ff_list path based on TRAIN
IMG_PATH = NEW_ROOT+"/train_images/" if TRAIN else NEW_ROOT+"/val_images/"
ANNOTATION_PATH_FULL = NEW_ROOT+"/annotations/train_full.json" if TRAIN else NEW_ROOT+"/annotations/val_full.json"
ANNOTATION_PATH_SIMPLE = NEW_ROOT+"/annotations/train_simple.json" if TRAIN else NEW_ROOT+"/annotations/val_simple.json"
FF_LIST_PATH = NEW_ROOT+"/meta/ff_train.txt" if TRAIN else NEW_ROOT+"/meta/ff_val.txt"
DEGREES = [0, 10, 90, 180, 270] if TRAIN else [0]

# Create the Class # -> Class Label Dictionary
class_names_LUT_full = {}
with open('xview_class_labels.txt') as f:
    for row in csv.reader(f):
        class_names_LUT_full[int(row[0].split(":")[0])] = row[0].split(":")[1]

# Create the Class # -> Class Label Dictionary
class_names_LUT_simple = {}
with open('xview_class_labels_simple.txt') as f:
    for row in csv.reader(f):
        class_names_LUT_simple[int(row[0].split(":")[0])] = row[0].split(":")[1]


# Load all of the labels from .geojson
all_coords, all_chips, all_classes = wv.get_labels(OLD_ROOT + "/xView_train.geojson")

print("Full Dataset Stats:")
print(all_coords.shape)
print(all_chips.shape)
print(all_classes.shape)


# Create directories if they don't exist
if not os.path.isdir(NEW_ROOT+"/annotations"):
    os.makedirs(NEW_ROOT+"/annotations")
if not os.path.isdir(IMG_PATH):
    os.makedirs(IMG_PATH)

# Get all of the unique .tif names from all_chips
#tif_names = np.unique(all_chips)


tif_names = [line.rstrip('\n') for line in open(FF_LIST_PATH)]

print("tif_names:", tif_names)


# Initialize JSON_Dict object
json_dict_full = xView_JSON_Dict(class_names_LUT_full)
json_dict_simple = xView_JSON_Dict(class_names_LUT_simple)


# For each unique .tif
for tif_idx, unique_tif in enumerate(tif_names):

    print("Working on: [{} / {}] {} ".format(tif_idx+1, len(tif_names), unique_tif))

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
    c_img, c_box, c_cls, _ = wv.chip_image_overlap(img=arr, coords=ff_coords, classes=ff_classes, shape=CHIP_SHAPE, overlap=OVERLAP)
    num_chips = len(c_img)
    print("\tNum Chips: ",num_chips)

    # For each image chip (i in range(num_chips))
    for i in range(num_chips):  

        print("\t\tChip #: ",i)

        # Calculate the center of the chip
        center = (int(c_img[i].shape[0]/2),int(c_img[i].shape[1]/2))

        # For each of the desired rotation degrees
        for deg in DEGREES:

            # Rotate the original chip and get the updated image/boxes/classes      
            tmp_img,tmp_box,tmp_cls = aug.rotate_image_and_boxes(c_img[i], deg, center, c_box[i], c_cls[i])

            # Git rid of very small boxes that are artifacts of chipping
            final_boxes = []
            final_classes = []
            final_classes_simple = []

            # Clip boxes correctly!!
            clipped_boxes = []
            for box in tmp_box:
                xMin,yMin,xMax,yMax = box
                xMin = clip(xMin, 0, CHIP_SHAPE[0]-1)
                yMin = clip(yMin, 0, CHIP_SHAPE[1]-1)
                xMax = clip(xMax, 0, CHIP_SHAPE[0]-1)
                yMax = clip(yMax, 0, CHIP_SHAPE[1]-1)
                clipped_boxes.append([xMin, yMin, xMax, yMax])
                

            # Eliminate clipped boxes whose area is too small
            for j,box in enumerate(clipped_boxes):
                xMin,yMin,xMax,yMax = box
                box_area = (xMax-xMin)*(yMax-yMin)
                box_width = xMax - xMin + 1
                box_height = yMax - yMin + 1
                if box_area > BOX_AREA_THRESH:
                    # Excludes odd error cases
                    if tmp_cls[j] in class_names_LUT_full.keys():        
                        # COCO expects boxes in [top-left x, top-left y, w, h] format
                        final_boxes.append([int(xMin), int(yMin), int(box_width), int(box_height)])
                        final_classes.append(int(tmp_cls[j]))
            
            # Create simple class annotations
            final_boxes_simple, final_classes_simple = convert_full_to_simple(final_boxes, final_classes)

                
            # Construct the saved chip name
            chip_name = "img_{}_{}_rot{}.jpg".format(unique_tif.split(".")[0], i, deg)

            # TIME TO WRITE**
            # First, check that the files do not already exist before writing
            #if os.path.exists(NEW_ROOT+"/annotations/"+xml_name) and os.path.exists(NEW_ROOT+"/JPEGImages/"+chip_name):
            #    continue
            
            # Convert the integer class labels to english labels
            #final_english_classes = [class_names_LUT_full[lbl] for lbl in final_classes]
            assert(len(final_boxes) == len(final_classes))

            # Add to annotation dictionary
            json_dict_full.add_image(chip_name, CHIP_SHAPE[1], CHIP_SHAPE[0], final_boxes, final_classes)
            json_dict_simple.add_image(chip_name, CHIP_SHAPE[1], CHIP_SHAPE[0], final_boxes, final_classes_simple)

            # Save the chipped image to disk
            #skimage.io.imsave(IMG_PATH+chip_name, tmp_img) #, quality=100)   

# The last step is to write the final JSON_Dict to json file
json_dict_full.write_file(ANNOTATION_PATH_FULL, indent=None) 
json_dict_simple.write_file(ANNOTATION_PATH_SIMPLE, indent=None) 
