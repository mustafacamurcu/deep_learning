import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
from PIL import Image
import matplotlib.image as mpimg
import glob

root = "/afs/csail.mit.edu/u/k/kocabey/"

BATCH_SIZE = 128

def caltech_random_slice_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/RSTrain/Points/*.txt') )
        jpg = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/RSTrain/Images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/RSTest/Points/*.txt') )
        jpg = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/RSTest/Images/*.jpg') )
    return jpg,txt

def bird_random_slice_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/BirdData/RSTrain/Points/*.txt') )
        jpg = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/BirdData/RSTrain/Images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/BirdData/RSTest/Points/*.txt') )
        jpg = sorted( glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/BirdData/RSTest/Images/*.jpg') )
    return jpg,txt

def MTFL_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/RSTrain/Points/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/RSTrain/Images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/SquareTest/Points/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/SquareTest/Images/*.jpg') )
    return jpg,txt

def LFPW_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/LFPWData/Train/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/LFPWData/Train/images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/LFPWData/Test/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/LFPWData/Test/images/*.jpg') )
    return jpg,txt

def Helen_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HelenData/Train/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HelenData/Train/images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HelenData/Test/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HelenData/Test/images/*.jpg') )
    return jpg,txt

def _300W_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/300WData/Train/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/300WData/Train/images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/300WData/Test/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/300WData/Test/images/*.jpg') )
    return jpg,txt

def COFW_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/COFWData/Train/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/COFWData/Train/images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/COFWData/Test/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/COFWData/Test/images/*.jpg') )
    return jpg,txt

def human_random_slice_directories(name):
    if name == "train":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HumanData/Train/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HumanData/Train/images/*.jpg') )
    if name == "test":
        txt = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HumanData/Test/annotations/*.txt') )
        jpg = sorted( glob.glob('/data/vision/torralba/health-habits/other/enes/HumanData/Test/images/*.jpg') )
    return jpg,txt

def import_human_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = []
        for j in range(len(data)):
            t2 = data[j].split()
            for k in range(3):
                t2[k] = float(t2[k])
            temp.append(t2)
        all_data.append( (jpg[i],temp) )
    return all_data

def import_caltech_point_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = list(data)
        for j in range(0,8):
            data[j] = int(temp[j])
        datam = []
        datam.append(data)
        directory = jpg[i]
        all_data.append( (directory , datam) )
    return all_data

def import_bird_point_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        show_progress(i,len(txt))
        f = open(txt[i],"r")
        data = f.readlines()
        temp = list(data)
        for j in range(0,45):
            data[j] = int(temp[j])
        datam = []
        datam.append(data)
        directory = jpg[i]
        all_data.append( (directory , datam) )
    print "\n"
    return all_data

def import_MTFL_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    print "loading data..."
    for i in range(len(txt)):
        show_progress(i,len(txt))
        f = open(txt[i],"r")
        data = f.readlines()
        temp = list(data)
        for j in range(0,10):
            data[j] = int(temp[j])
        datam = []
        datam.append(data)
        directory = jpg[i]
        all_data.append( (directory , datam) )
    print "\n"
    return all_data

def import_LFPW_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = []
        for j in range(len(data)):
            t2 = data[j].split()
            for k in range(2):
                t2[k] = float(t2[k])
            temp.append(t2)
        all_data.append( (jpg[i],temp) )
    return all_data

def import_Helen_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = []
        for j in range(len(data)):
            t2 = data[j].split()
            for k in range(2):
                t2[k] = float(t2[k])
            temp.append(t2)
        all_data.append( (jpg[i],temp) )
    return all_data

def import_300W_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = []
        for j in range(len(data)):
            t2 = data[j].split()
            for k in range(2):
                t2[k] = float(t2[k])
            temp.append(t2)
        all_data.append( (jpg[i],temp) )
    return all_data

def import_COFW_data(jpg,txt):
    print len(jpg), " ", len(txt)
    all_data = []
    for i in range(len(txt)):
        f = open(txt[i],"r")
        data = f.readlines()
        temp = []
        for j in range(len(data)):
            t2 = data[j].split()
            for k in range(2):
                t2[k] = float(t2[k])
            temp.append(t2)
        all_data.append( (jpg[i],temp) )
    return all_data


def show_progress(k,N):
    num_bars = 100
    u = (k * num_bars) / N
    a = "["
    for i in range(num_bars):
        if i <= u:
            a += "->"
        else:
            a += "  "
    a += "]"
    sys.stdout.write('\r\x1b[K')
    sys.stdout.write(a)
    sys.stdout.flush()

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

def get_next_trn_batch_face_point_conv5_convolution(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,4))
    batch_point_y = np.zeros((BATCH_SIZE,4))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(4):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 2*j ] * 9 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][2*j+1] * 9 / float(aa)

    return (batch_x,batch_point_x,batch_point_y)

def get_next_trn_batch_MTFL(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,5))
    batch_point_y = np.zeros((BATCH_SIZE,5))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(5):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 2*j ] * 9 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][2*j+1] * 9 / float(aa)

    return (batch_x,batch_point_x,batch_point_y)

def get_next_trn_batch_bird(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_point_x = np.zeros((BATCH_SIZE,15))
    batch_point_y = np.zeros((BATCH_SIZE,15))
    batch_existence = np.zeros((BATCH_SIZE,15))

    for i in range(BATCH_SIZE):
        (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
        for j in range(15):
            batch_point_x[i][j] = all_data[trn_indices[i]][1][0][ 3*j ] * 9 / float(aa)
            batch_point_y[i][j] = all_data[trn_indices[i]][1][0][3*j+1] * 9 / float(aa)
            batch_existence[i][j] = all_data[trn_indices[i]][1][0][3*j+2]

    return (batch_x,batch_point_x,batch_point_y,batch_existence)
