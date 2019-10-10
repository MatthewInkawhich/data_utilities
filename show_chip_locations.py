from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import csv
import os
import argparse
import cv2


def get_image(fname):
    return np.array(Image.open(fname))

def draw_chips_overlap(img, shape=(600,600), overlap=0.2):
    height,width,_ = img.shape
    wn,hn = shape
    
    # Initialize numpy array to fill with chip image arrs
    x_offsets = []
    y_offsets = []
    last_edge = 0
    i = 0
    while (i+wn) < width:
        x_offsets.append(i)
        last_edge = i+wn
        i += int(wn * (1 - overlap))
    # handle end case for x
    if last_edge < width - 1:
        # In this case, there is a strip at the right edge that did not get considered
        x_offsets.append(width - wn)

    last_edge = 0
    j = 0
    while (j+hn) < height:
        y_offsets.append(j)
        last_edge = j+hn
        j += int(hn * (1 - overlap))
    # handle end case for y
    if last_edge < height - 1:
        # In this case, there is a strip at the right edge that did not get considered
        y_offsets.append(height - hn)
    
    # k is a count of each chip
    k = 0
    for i in x_offsets:
        for j in y_offsets:
            # Calculate coordinate to place text
            x_text = i + (shape[0]//2) - 15
            y_text = j + (shape[0]//2) + 15
            cv2.putText(img, str(k), (x_text, y_text), cv2.FONT_HERSHEY_PLAIN, 10.0, (0, 255, 0), thickness=10)
            k += 1
     
    return img



#####################################################################
### MAIN
#####################################################################

# Read argument
parser = argparse.ArgumentParser(description='Show full frame image with indications of where each chip is located')
parser.add_argument('image_id', type=int)
args = parser.parse_args()

# Form image path
image_path = "../xView-meta/train_images/" + str(args.image_id) + ".tif"
img = get_image(image_path)
dimg = draw_chips_overlap(img)

plt.imshow(dimg)
plt.show()

