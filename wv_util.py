"""
Copyright 2018 Defense Innovation Unit Experimental
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from PIL import Image
import numpy as np
import json
from tqdm import tqdm

"""
xView processing helper functions for use in data processing.
"""

def scale(x,range1=(0,0),range2=(0,0)):
    """
    Linear scaling for a value x
    """
    return range2[0]*(1 - (x-range1[0]) / (range1[1]-range1[0])) + range2[1]*((x-range1[0]) / (range1[1]-range1[0]))


def get_image(fname):    
    """
    Get an image from a filepath in ndarray format
    """
    return np.array(Image.open(fname))


def get_labels(fname):
    """
    Gets label data from a geojson label file

    Args:
        fname: file path to an xView geojson label file

    Output:
        Returns three arrays: coords, chips, and classes corresponding to the
            coordinates, file-names, and classes for each ground truth.
    """
    with open(fname) as f:
        data = json.load(f)

    coords = np.zeros((len(data['features']),4))
    chips = np.zeros((len(data['features'])),dtype="object")
    classes = np.zeros((len(data['features'])))

    for i in tqdm(range(len(data['features']))):
        if data['features'][i]['properties']['bounds_imcoords'] != []:
            b_id = data['features'][i]['properties']['image_id']
            val = np.array([int(num) for num in data['features'][i]['properties']['bounds_imcoords'].split(",")])
            chips[i] = b_id
            classes[i] = data['features'][i]['properties']['type_id']
            if val.shape[0] != 4:
                print("Issues at %d!" % i)
            else:
                coords[i] = val
        else:
            chips[i] = 'None'

    return coords, chips, classes


def boxes_from_coords(coords):
    """
    Processes a coordinate array from a geojson into (xmin,ymin,xmax,ymax) format

    Args:
        coords: an array of bounding box coordinates

    Output:
        Returns an array of shape (N,4) with coordinates in proper format
    """
    nc = np.zeros((coords.shape[0],4))
    for ind in range(coords.shape[0]):
        x1,x2 = coords[ind,:,0].min(),coords[ind,:,0].max()
        y1,y2 = coords[ind,:,1].min(),coords[ind,:,1].max()
        nc[ind] = [x1,y1,x2,y2]
    return nc


def chip_image(img,coords,classes,shape=(300,300)):
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
    height,width,_ = img.shape
    wn,hn = shape
    
    w_num,h_num = (int(width/wn),int(height/hn))
    images = np.zeros((w_num*h_num,hn,wn,3))
    total_boxes = {}
    total_classes = {}
    
    k = 0
    for i in range(w_num):
        for j in range(h_num):
            x = np.logical_or( np.logical_and((coords[:,0]<((i+1)*wn)),(coords[:,0]>(i*wn))),
                               np.logical_and((coords[:,2]<((i+1)*wn)),(coords[:,2]>(i*wn))))
            out = coords[x]
            y = np.logical_or( np.logical_and((out[:,1]<((j+1)*hn)),(out[:,1]>(j*hn))),
                               np.logical_and((out[:,3]<((j+1)*hn)),(out[:,3]>(j*hn))))
            outn = out[y]
            out = np.transpose(np.vstack((np.clip(outn[:,0]-(wn*i),0,wn),
                                          np.clip(outn[:,1]-(hn*j),0,hn),
                                          np.clip(outn[:,2]-(wn*i),0,wn),
                                          np.clip(outn[:,3]-(hn*j),0,hn))))
            box_classes = classes[x][y]
            
            if out.shape[0] != 0:
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            chip = img[hn*j:hn*(j+1),wn*i:wn*(i+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8),total_boxes,total_classes


# With overlaps
def chip_image_overlap(img,coords,classes,shape=(300,300),overlap=0.2):
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


    w_num = len(x_offsets)
    h_num = len(y_offsets)

    #print("width:", width)
    #print("height:", height)
    #print("wn:", wn)
    #print("hn:", hn)
    #print("x_offsets:", x_offsets)
    #print("y_offsets:", y_offsets)
    #exit()

    images = np.zeros((w_num*h_num,hn,wn,3))
    
    # Initialize dicts
    total_boxes = {}
    total_classes = {}

    # keep track of x and y offsets of each chip for stitching later
    offsets = []
    
    # k is a count of each chip
    k = 0
    for i in x_offsets:
        for j in y_offsets:
            # Track boxes that fall within x-range of this chip
            min_x = i
            max_x = min_x+wn
            x = np.logical_or( np.logical_and((coords[:,0]<max_x),(coords[:,0]>min_x)),
                               np.logical_and((coords[:,2]<max_x),(coords[:,2]>min_x)))
            out = coords[x]

            # Track boxes that fall within y-range of this chip
            min_y = j
            max_y = min_y+hn
            y = np.logical_or( np.logical_and((out[:,1]<max_y),(out[:,1]>min_y)),
                               np.logical_and((out[:,3]<max_y),(out[:,3]>min_y)))
            outn = out[y]
            
            # Make bbox coords relative to chip (not ff image) and clip all boxes within clip size bounds
            out = np.transpose(np.vstack((np.clip(outn[:,0]-min_x,0,wn-1),
                                          np.clip(outn[:,1]-min_y,0,hn-1),
                                          np.clip(outn[:,2]-min_x,0,wn-1),
                                          np.clip(outn[:,3]-min_y,0,hn-1))))

            # Get classes for the boxes that are in this chip
            box_classes = classes[x][y]
            
            # If there are objects in this chip
            if out.shape[0] != 0:
                # Add boxes and chips to corresponding dicts
                total_boxes[k] = out
                total_classes[k] = box_classes
            else:
                # Else, there are no objects in this chip
                total_boxes[k] = np.array([[0,0,0,0]])
                total_classes[k] = np.array([0])
            
            # Chip actual image array
            chip = img[min_y:max_y, min_x:max_x,:3]
            images[k]=chip
            
            k += 1

            offsets.append([min_x, min_y])

    return images.astype(np.uint8),total_boxes,total_classes,offsets
