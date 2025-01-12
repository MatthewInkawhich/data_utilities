import os
import math
import numpy as np
import random
import time
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import wv_util as wv
import aug_util as aug


chip_size = 200
ff_name = '../xView/train_images/104.tif'
#ff_name = '../xView/train_images/2322.tif'
#ff_name = '../xView/train_images/1211.tif'
#ff_name = '../xView/train_images/1418.tif'
ff_arr = wv.get_image(ff_name)

grid_color = [255, 0, 255]
print(ff_arr.shape)
for i in range(0, ff_arr.shape[1], chip_size):
    print(i)
    ff_arr[:,i:i+10,:] = grid_color
for i in range(0, ff_arr.shape[0], chip_size):
    print(i)
    ff_arr[i:i+10,:,:] = grid_color


#ff_arr[:,::chip_size,:] = grid_color
#ff_arr[::chip_size,:,:] = grid_color

plt.title('Img {}: {}x{} Chips'.format(ff_name.split('/')[-1].split('.')[0], chip_size, chip_size))
plt.axis('off')
plt.imshow(ff_arr)
plt.savefig('../../xview_project/chip{}.pdf'.format(chip_size))
plt.show()

"""
all_coords, all_ffs, all_classes = wv.get_labels('../xView/xView_train.geojson')

ff_coords = all_coords[all_ffs==ff_name.split('/')[-1]]
ff_classes = all_classes[all_ffs==ff_name.split('/')[-1]].astype(np.int64)

#print("ff_coords:", ff_coords)
#print("ff_classes:", ff_classes)

c_img, c_box, c_cls= wv.chip_image_overlap(img=ff_arr, coords=ff_coords, classes=ff_classes, shape=(chip_size,chip_size), overlap=.20)
c_img1, c_box1, c_cls1 = wv.chip_image(img=ff_arr, coords=ff_coords, classes=ff_classes, shape=(chip_size,chip_size))


print("c_img1:", c_img1.shape)
print("c_img:", c_img.shape)
#print("c_box:", c_box)
#print("c_cls:", c_cls)

n = 3
fig, ax = plt.subplots(n,2)

for i in range(n):
    labeled = aug.draw_bboxes(c_img1[i], c_box1[i])
    ax[i,0].imshow(labeled)

for i in range(n):
    labeled = aug.draw_bboxes(c_img[i], c_box[i])
    ax[i,1].imshow(labeled)

plt.show()
"""
