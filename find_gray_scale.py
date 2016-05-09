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

Face_jpg_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Images/')
Face_txt_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Annotations/')
z = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/serroimage.txt","w")
k = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/serroannotation.txt","w")
for i in range(1,len(Face_jpg_directories)):
	if i % 10 == 0:
		print i
	directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Images/' + Face_jpg_directories[i]
	img = imread(directory)
	if len(img.shape) < 3:
		z.write(directory + "\n" )
		k.write("/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Annotations/" + Face_txt_directories[i-1] + "\n")
