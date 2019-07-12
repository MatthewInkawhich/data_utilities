# This script is to be called after creating the JPEGImages and Annotations.
#  Here, we create the ImageSets/Main/train.txt val.txt. Since xView only has
#  labels for the training set, we will split the train set into a train and
#  val set. Here we randomly sample from the full train set to create them.

import os
import glob
import random
import sys

# Directory to write train.txt and val.txt files to
IMAGESETS_DIR = "/bigdata/NFTI/datasets/xView2018/xView-voc/ImageSets/Main"
ANNOTATIONS_DIR = "/bigdata/NFTI/datasets/xView2018/xView-voc/Annotations"
IMAGES_DIR = "/bigdata/NFTI/datasets/xView2018/xView-voc/JPEGImages"
#IMAGESETS_DIR = "/raid/inkawhna/WORK/xView-voc/ImageSets/Main"
#ANNOTATIONS_DIR = "/raid/inkawhna/WORK/xView2018/VOCdevkit/VOC2007/Annotations"
#IMAGESETS_DIR = "/raid/inkawhna/WORK/xView-voc/ImageSets/Main"
#ANNOTATIONS_DIR = "/raid/inkawhna/WORK/xView-voc/Annotations"

train_file = open(IMAGESETS_DIR+"/train.txt","w")
val_file = open(IMAGESETS_DIR+"/val.txt","w")

# read in all filenames from annotations dir into list
print("Reading in all filenames...")
files = glob.glob(ANNOTATIONS_DIR+"/*.xml")
num_files = len(files)
print("Read in {} files".format(num_files))
files = [f.split("/")[-1][:-4] for f in files]


# Save full train list incase we need it later
print("Saving full train file...")
fulltrain = open(IMAGESETS_DIR+"/fulltrain.txt","w")
for fl in files:
	fulltrain.write(fl+"\n")
fulltrain.close()

# Dont want img_999_1_rot10 in the training set and img_999_1_rot20 in the 
#   validation set as it may artifically inflate the mAP. Handle that here.

print("Finding unique prefixes...")
# create a unique list of the prefixes [i.e. img_999_1_500x500]
unique_set = set()
for f in files:
	pieces = f.split("_")
	assert(len(pieces)==5)
	prefix = "{}_{}_{}_{}".format(pieces[0],pieces[1],pieces[2],pieces[3])
	#if prefix not in unique_list:
	#	unique_list.append(prefix)
	unique_set.add(prefix)
print("Found {} unique prefixes".format(len(unique_set)))

unique_list = list(unique_set)


# shuffle list
random.shuffle(unique_list)
# create sample index array of 500 indexes of which will make up the val set
val_idxs = random.sample(range(len(unique_list)),500)

print("Collecting and writing all augmentations...")
train_cnt = 0
val_cnt = 0
for i in range(len(unique_list)):
	sys.stdout.write("{} / {}\r".format(i,len(unique_list)))
	sys.stdout.flush()
	# Get the prefix	
	prefix = unique_list[i]+"_"
	if i in val_idxs:
		# Here we are only adding the rot0. Must first ASSERT that it exists
		fname = ANNOTATIONS_DIR+"/"+prefix+"rot0"
		if (os.path.isfile(fname+".xml")) and (os.path.isfile(IMAGES_DIR+"/"+prefix+"rot0.png")):
		  # Only add rot0 to validation list. Wrong assumption, not always rot0
		  val_file.write(prefix+"rot0\n")
		  val_cnt += 1
	else:
		# Old and slow
		# find all the files with this prefix
		#augs = glob.glob(ANNOTATIONS_DIR+"/"+prefix+"*")
		# Strip augs down to the basenames [i.e. img_999_1_rot10] 
		#augs = [f.split("/")[-1][:-4] for f in augs]

		# Assume all augs exist
		augs = []
		for b in range(0,360,10):
			augs.append("{}rot{}".format(prefix,b))

		# Write all augmentations to train list
		for a in augs:
			if os.path.isfile(ANNOTATIONS_DIR + "/" + a + ".xml") and os.path.isfile(IMAGES_DIR + "/" + a + ".png") :
				train_file.write(a+"\n")
				train_cnt += 1
	#if i == 10:
	#	break


print("Wrote {} train and {} val files".format(train_cnt, val_cnt))

exit()


##### OLD CODE BELOW FOR USE WITH NO AUGMENTATIONS
# shuffle list
random.shuffle(files)

# create sample index array of 500 indexes of which will make up the val set
val_idxs = random.sample(range(num_files),500)

train_cnt = 0
val_cnt = 0

# Write train and val lists according to sampled indexes
for i in range(num_files):
	if i in val_idxs:
		val_file.write(files[i]+"\n")
		val_cnt += 1
	else:
		train_file.write(files[i]+"\n")
		train_cnt += 1

print("Wrote {} train and {} val files".format(train_cnt, val_cnt))
