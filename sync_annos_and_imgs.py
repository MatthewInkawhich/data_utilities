import os
import glob

ANNOS_DIR = "/bigdata/NFTI/datasets/xView2018/xView-voc/Annotations"
IMGS_DIR = "/bigdata/NFTI/datasets/xView2018/xView-voc/JPEGImages"

# Cycle through annos and search for lost jpgs
for anno in glob.glob(ANNOS_DIR+"/*.xml"):
	fname = anno.split("/")[-1][:-4]
	if not os.path.isfile(IMGS_DIR+"/"+fname+".png"):
		print("Missing: ", fname)

# Cycle through jpgs and search for lost annos


