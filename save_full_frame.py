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


ff_name = '../xView/train_images/100.tif'
ff_id = ff_name.split('/')[-1].split('.')[0]
ff_arr = wv.get_image(ff_name)

plt.axis('off')
plt.imshow(ff_arr)
plt.savefig('../../xview_project/images/ff/ff_{}.png'.format(ff_id))
plt.show()

