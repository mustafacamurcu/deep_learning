import tensorflow as tf
import numpy as np
import scipy.misc
from PIL import Image
import matplotlib.image as mpimg
import glob
import random

BATCH_SIZE = 10

def imread(path):
    return scipy.misc.imread(path).astype(np.float32)

def image_preprocess(path):

    image = imread(path)
    image = image[:, :, [2,1,0]]

    new_shape = (224, 224)
    image = scipy.misc.imresize(image, new_shape)

    image = np.array(image, dtype = np.float32)
    image[:,:,0] = image[:,:,0] - 104;
    image[:,:,1] = image[:,:,1] - 117;
    image[:,:,2] = image[:,:,2] - 123;

    return image

def get_next_trn_batch_bird(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,224,224,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,15))
    batch_point_y = np.zeros((BATCH_SIZE,15))
    batch_existence = np.zeros((BATCH_SIZE,15))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(15):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 3*j ] * 10 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][3*j+1] * 10 / float(aa)
            batch_existence[i][j] = all_data[trn_indices[i]][1][0][3*j+2]

    return (batch_x,batch_point_x,batch_point_y,batch_existence)

def get_next_trn_batch_multilayer_bird(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,224,224,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,15))
    batch_point_y = np.zeros((BATCH_SIZE,15))
    batch_existence = np.zeros((BATCH_SIZE,15))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(15):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 3*j ] / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][3*j+1] / float(aa)
            batch_existence[i][j] = all_data[trn_indices[i]][1][0][3*j+2]

    return (batch_x,batch_point_x,batch_point_y,batch_existence)

def get_next_trn_batch_face(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))
    batch_x = np.zeros((BATCH_SIZE,224,224,3))

    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )
    batch_point_x = np.zeros((BATCH_SIZE,5))
    batch_point_y = np.zeros((BATCH_SIZE,5))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(5):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 2*j ] * 10 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][2*j+1] * 10 / float(aa)

    return (batch_x,batch_point_x,batch_point_y)

def get_next_trn_batch_scratch_face(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))
    batch_x = np.zeros((BATCH_SIZE,224,224,3))

    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )
    batch_point_x = np.zeros((BATCH_SIZE,5))
    batch_point_y = np.zeros((BATCH_SIZE,5))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(5):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 2*j ] * 42 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][2*j+1] * 42 / float(aa)

    return (batch_x,batch_point_x,batch_point_y)

def get_next_trn_caltech_batch_face(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))
    batch_x = np.zeros((BATCH_SIZE,224,224,3))

    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )
    batch_point_x = np.zeros((BATCH_SIZE,4))
    batch_point_y = np.zeros((BATCH_SIZE,4))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(4):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 2*j ] * 10 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][2*j+1] * 10 / float(aa)

    return (batch_x,batch_point_x,batch_point_y)

def get_next_trn_batch_human(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,224,224,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,14))
    batch_point_y = np.zeros((BATCH_SIZE,14))
    batch_existence = np.zeros((BATCH_SIZE,14))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(14):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][j][0] * 10 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][j][1] * 10 / float(aa)
            batch_existence[i][j] = 1.

    return (batch_x,batch_point_x,batch_point_y,batch_existence)
