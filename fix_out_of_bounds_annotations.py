# Run this script after creating JPEGImages and Annotations to remove
# excessively cutoff bboxes.
import glob
#import xml.etree.ElementTree as ET
from lxml import etree as ET
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from collections import defaultdict
import math


annotation_dir = os.path.join(os.path.expanduser('~'), 'WORK', 'data', 'xView-voc-700', 'Annotations')
annotation_names = glob.glob(annotation_dir + '/*.xml')


def clip(x, minimum=0, maximum=699):
    if x < minimum:
        return minimum
    elif x > maximum:
        return maximum
    else:
        return x

for fn in annotation_names:
    tree = ET.parse(fn)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        bbox.find('xmin').text = str(clip(int(bbox.find('xmin').text)))
        bbox.find('ymin').text = str(clip(int(bbox.find('ymin').text)))
        bbox.find('xmax').text = str(clip(int(bbox.find('xmax').text)))
        bbox.find('ymax').text = str(clip(int(bbox.find('ymax').text)))
        
        # Remove mostly cut-off boxes from edges
        width = int(bbox.find('xmax').text) - int(bbox.find('xmin').text)
        height = int(bbox.find('ymax').text) - int(bbox.find('ymin').text)
        if width < 5 or height < 5:
            obj.getparent().remove(obj)

    tree.write(fn.replace('Annotations/', 'Annotations_new/'))