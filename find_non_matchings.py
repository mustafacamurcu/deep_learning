import numpy as np
import sys
import scipy.misc
import math
import time
import random
import detection

from os import listdir

def imread(path):
    return scipy.misc.imread(path).astype(np.float32)

def image_preprocess(path):

    image = imread(path)
    new_shape = (227, 227)
    image = scipy.misc.imresize(image, new_shape)

    image = np.array(image, dtype = np.float32)
    image[:,:,0] = image[:,:,0] - 123;
    image[:,:,1] = image[:,:,1] - 117;
    image[:,:,2] = image[:,:,2] - 104;

    return image

Face_jpg_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/JPEGImages/')
Face_xml_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/Annotations')

z = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/serro.txt","w")
for i in range(1,len(Face_xml_directories)):
	a = False
	for j in range(1, len(Face_jpg_directories)):
		if Face_jpg_directories[j][:-4] == Face_xml_directories[i][:-3]:
			a = True
	if a == False:
		z.write('/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/Annotations/' + Face_xml_directories[i] + "\n")
