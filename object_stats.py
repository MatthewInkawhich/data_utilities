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


label_file_path = '../xView-meta/xview_class_labels_clean.txt'
labels = [line.rstrip('\n').lower() for line in open(label_file_path)]

annotation_dir = os.path.join(os.path.expanduser('~'), 'WORK', 'data', 'xView-voc-ff', 'Annotations')
annotation_names = glob.glob(annotation_dir + '/*.xml')

stat = defaultdict(lambda: defaultdict(int))

isqrt = lambda x: int(math.sqrt(x))

min_objs = 1000000
max_objs = 0
min_objs_fn = ""
max_objs_fn = ""
total_objs = 0
for fn in annotation_names:
    tree = ET.parse(fn)
    objs = tree.findall('object')
    total_objs += len(objs)
    if len(objs) < min_objs:
        min_objs = len(objs)
        min_objs_fn = fn.split('/')[-1]
    if len(objs) > max_objs:
        max_objs = len(objs)
        max_objs_fn = fn.split('/')[-1]
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
        # Add to stat
        stat[name]['count'] += 1
        stat[name]['sum'] += area
        if stat[name]['min'] == 0 or area < stat[name]['min']:
            stat[name]['min'] = area
        if area > stat[name]['max']:
            stat[name]['max'] = area


# Print stats
with open('object_stats.csv', mode='w') as csv_file:
    #fieldnames = ['class', 'avg', 'min', 'max']
    fieldnames = ['avg', 'min', 'max']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for key in labels:
        #writer.writerow({'class': key, 'avg': isqrt(stat[key]['sum'] // stat[key]['count']), 'min': isqrt(stat[key]['min']), 'max': isqrt(stat[key]['max'])})
        writer.writerow({'avg': isqrt(stat[key]['sum'] // stat[key]['count']), 'min': isqrt(stat[key]['min']), 'max': isqrt(stat[key]['max'])})
        print("key:", key)
        print("avg:", stat[key]['sum'] // stat[key]['count'], int(math.sqrt(stat[key]['sum'] // stat[key]['count'])))
        print("min:", stat[key]['min'], int(math.sqrt(stat[key]['min'])))
        print("max:", stat[key]['max'], int(math.sqrt(stat[key]['max'])))
        print("\n")
        

print(len(stat))

print("Min objs:", min_objs_fn, min_objs)
print("Max objs:", max_objs_fn, max_objs)
print("Avg objs: {} / {} = {}".format(total_objs, len(annotation_names), total_objs/len(annotation_names)))

