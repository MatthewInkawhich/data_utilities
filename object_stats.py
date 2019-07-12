import glob
import xml.etree.ElementTree as ET
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

stat = defaultdict(lambda: defaultdict(int))

isqrt = lambda x: int(math.sqrt(x))

for fn in annotation_names:
    tree = ET.parse(fn)
    objs = tree.findall('object')
    for ix, obj in enumerate(objs):
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        area = width * height
        if x2-x1 <= 0:
            print("x2-x1 <= 0", "x2:", x2, "x1:", x1)
        if y2-y1 <= 0:
            print("y2-y1 <= 0", "y2:", y2, "y1:", y1)
        
        if x1 < 0 or x1 >= 700:
            print("x1 out of bounds:", x1)
        if x2 < 0 or x2 >= 700:
            print("x2 out of bounds:", x2)
            print(fn)
        if y1 < 0 or y1 >= 700:
            print("y1 out of bounds:", y1)
        if y2 < 0 or y2 >= 700:
            print("y2 out of bounds:", y2)
"""
        # Add to stat
        stat[name]['count'] += 1
        stat[name]['sum'] += area
        if stat[name]['min'] == 0 or area < stat[name]['min']:
            stat[name]['min'] = area
        if area > stat[name]['max']:
            stat[name]['max'] = area


# Print stats
with open('object_stats.csv', mode='w') as csv_file:
    fieldnames = ['class', 'avg', 'min', 'max']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key, _ in stat.items():
        writer.writerow({'class': key, 'avg': isqrt(stat[key]['sum'] // stat[key]['count']), 'min': isqrt(stat[key]['min']), 'max': isqrt(stat[key]['max'])})
        print("key:", key)
        print("avg:", stat[key]['sum'] // stat[key]['count'], int(math.sqrt(stat[key]['sum'] // stat[key]['count'])))
        print("min:", stat[key]['min'], int(math.sqrt(stat[key]['min'])))
        print("max:", stat[key]['max'], int(math.sqrt(stat[key]['max'])))
        print("\n")
        

print(len(stat))
"""


