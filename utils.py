import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
import detection
from PIL import Image
import matplotlib.image as mpimg

from os import listdir
import glob
import xml.etree.ElementTree as ET

root = "/afs/csail.mit.edu/u/k/kocabey/"

Caltech_txt_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Annotations')
Caltech_jpg_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Images')

BATCH_SIZE = 128
FACE_TEST_SIZE = 500
CIRCLE_TEST_SIZE = 1000
CALTECH_TEST_SIZE = 1000
CTR = 0
VOC_NAMES = ['person','bird','cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', \
             'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable',\
              'pottedplant','sofa', 'tvmonitor']

SAMPLE_VOC_NAMES = ['cat']

def caltech_get_directories(name):
    if name == "train":
        txt = []; jpg = [];
        for i in range(3):
            for j in range(5):
                txt += \
                glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/NewTrain/NewPoints/' + str(i) + '/' + str(j) + '/*.txt')
                jpg += \
                glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/NewTrain/NewImages/' + str(i) + '/' + str(j) + '/*.jpg')
    if name == "test":
        txt = []; jpg = [];
        for i in range(3):
            for j in range(5):
                txt += \
                glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/NewTest/NewPoints/' + str(i) + '/' + str(j) + '/*.txt')
                jpg += \
                glob.glob('/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/NewTest/NewImages/' + str(i) + '/' + str(j) + '/*.jpg')
    return jpg,txt

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

def import_data(name):
    all_data = []
    for i in range( 1, len(VOC_xml_directories)):
        xml = VOC_xml_directories[i]
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit3/VOC2010/Annotations/' + xml
        tree = ET.parse(directory)
        root = tree.getroot()

        data = []

        for obj in root.findall( 'object' ):
            if obj.find('name').text == name:
                bndbox = obj.find( 'bndbox' )
                data.append( ( int(bndbox.find('xmin').text), int(bndbox.find('xmax').text),
                              int(bndbox.find('ymin').text), int(bndbox.find('ymax').text) ) )

        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/JPEGImages/' + VOC_jpg_directories[i]
        all_data.append( (directory , data) )
    return all_data

def import_face_data():
    all_data = []
    for i in range( FACE_TEST_SIZE + 1, len(Face_xml_directories)):
        xml = Face_xml_directories[i]
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/Annotations/' + xml
        tree = ET.parse(directory)
        root = tree.getroot()

        data = []
        data.append(  ( int(root.find('xmin').text), int(root.find('xmax').text), \
                       int(root.find('ymin').text), int(root.find('ymax').text) ) )

        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/JPEGImages/' + Face_jpg_directories[i]
        all_data.append( (directory , data) )
    return all_data

def import_circle_data():
    all_data = []
    for i in range( CIRCLE_TEST_SIZE + 1, len(Circle_txt_directories)):
        f = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Annotations/" + Circle_txt_directories[i],"r")
        data = f.readlines()
        temp = list(data)
        data[0] = int(temp[0])
        data[1] = int(temp[2])
        data[2] = int(temp[1])
        data[3] = int(temp[3])
        datam = []
        datam.append(data)
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Images/' + Circle_jpg_directories[i]
        all_data.append( (directory , datam) )
    return all_data

def import_caltech_data():
    all_data = []
    for i in range( CALTECH_TEST_SIZE + 1, len(Caltech_txt_directories)):
        f = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Annotations/" + Caltech_txt_directories[i],"r")
        data = f.readlines()
        temp = list(data)
        data[0] = int(temp[0])
        data[1] = int(temp[1])
        data[2] = int(temp[2])
        data[3] = int(temp[3])
        datam = []
        datam.append(data)
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Images/' + Caltech_jpg_directories[i]
        all_data.append( (directory , datam) )
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
        f = open(txt[i],"r")
        data = f.readlines()
        temp = list(data)
        for j in range(0,45):
            data[j] = int(temp[j])
        datam = []
        datam.append(data)
        directory = jpg[i]
        all_data.append( (directory , datam) )
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

def import_face_test_data():
    all_data = []
    for i in range( 1, FACE_TEST_SIZE + 1):
        xml = Face_xml_directories[i]
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/Annotations/' + xml
        tree = ET.parse(directory)
        root = tree.getroot()

        data =  ( int(root.find('xmin').text), int(root.find('xmax').text), \
                       int(root.find('ymin').text), int(root.find('ymax').text) )

        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/FaceData/JPEGImages/' + Face_jpg_directories[i]
        all_data.append( (directory , data) )
    return all_data

def import_circle_test_data():
    all_data = []
    for i in range( 1 ,CIRCLE_TEST_SIZE + 1):
        f = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Annotations/" + Circle_txt_directories[i],"r")
        data = f.readlines()
        temp = list(data)
        data[0] = int(temp[0])
        data[1] = int(temp[2])
        data[2] = int(temp[1])
        data[3] = int(temp[3])
        datam = []
        datam.append(data)
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Images/' + Circle_jpg_directories[i]
        all_data.append( (directory , datam) )
    return all_data

def import_caltech_test_data():
    all_data = []
    for i in range(1, CALTECH_TEST_SIZE+1):
        f = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Annotations/" + Caltech_txt_directories[i],"r")
        data = f.readlines()
        temp = list(data)
        data[0] = int(temp[0])
        data[1] = int(temp[1])
        data[2] = int(temp[2])
        data[3] = int(temp[3])
        datam = []
        datam.append(data)
        directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/CaltechData/Images/' + Caltech_jpg_directories[i]
        all_data.append( (directory , datam) )
    return all_data

def import_test_data():
    test_data = []
    test_name_data = []
    f = open('/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit/VOC2010/ImageSets/Main/test.txt')
    data = f.readlines()
    f.close()
    a = '/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit/VOC2010/JPEGImages/'
    for name in data:
        directory = a + name.rstrip() + '.jpg'
        test_data.append(directory)
        test_name_data.append(name.rstrip())

    return test_data,test_name_data

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

def transform_data(a, b, a_1, b_1, data):
    ratio_1 = a_1 / float(a)
    ratio_2 = b_1 / float(b)
    xmin = data[0] * ratio_1
    xmax = data[1] * ratio_1
    ymin = data[2] * ratio_2
    ymax = data[3] * ratio_2
    return ( xmin, xmax, ymin, ymax )

def inverse_transform_data(a, b, a_1, b_1, data):
    ratio_1 = a / float(a_1)
    ratio_2 = b / float(b_1)
    xmin = math.floor( data[0] * ratio_1 )
    xmax = math.ceil ( data[1] * ratio_1 )
    ymin = math.floor( data[2] * ratio_2 )
    ymax = math.ceil ( data[3] * ratio_2 )
    return ( xmin, xmax, ymin, ymax )

def pool5_data( data ):
    RATIO = 227 / float(6)
    xmin = math.floor ( data[0] / RATIO )
    xmax = math.ceil  ( data[1] / RATIO )
    ymin = math.floor ( data[2] / RATIO )
    ymax = math.ceil  ( data[3] / RATIO )
    return ( xmin, xmax, ymin, ymax )

def conv5_data( data ):
    RATIO = 227 / float(13)
    xmin = math.floor ( data[0] / RATIO )
    xmax = math.ceil  ( data[1] / RATIO )
    ymin = math.floor ( data[2] / RATIO )
    ymax = math.ceil  ( data[3] / RATIO )
    return ( xmin, xmax, ymin, ymax )

def inverse_pool5_data( data ):
    RATIO = 6 / float(227)
    xmin = data[0] / RATIO
    xmax = data[1] / RATIO
    ymin = data[2] / RATIO
    ymax = data[3] / RATIO
    return ( xmin, xmax, ymin, ymax )

def inverse_conv5_data( data ):
    RATIO = 13 / float(227)
    xmin = data[0] / RATIO
    xmax = data[1] / RATIO
    ymin = data[2] / RATIO
    ymax = data[3] / RATIO
    return ( xmin, xmax, ymin, ymax )

def intersection_over_union(data_1, data_2):
    (xmin_1,xmax_1,ymin_1,ymax_1) = data_1
    (xmin_2,xmax_2,ymin_2,ymax_2) = data_2
    s_intersection = max(0, min(xmax_1, xmax_2) - max(xmin_1, xmin_2)) * max(0, min(ymax_1, ymax_2) - max(ymin_1, ymin_2))
    s_data_1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
    s_data_2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)
    s_union = s_data_1 + s_data_2 - s_intersection

    return s_intersection / float(s_union)

def distance_between_centers( data_1, data_2 ):
    (xmin_1,xmax_1,ymin_1,ymax_1) = data_1
    (xmin_2,xmax_2,ymin_2,ymax_2) = data_2
    center_1 = ( (xmin_1 + xmax_1)/2.0,  (ymin_1 + ymax_1)/2.0 )
    center_2 = ( (xmin_2 + xmax_2)/2.0,  (ymin_2 + ymax_2)/2.0 )

    return math.sqrt((center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2)

def process_data_pool5(a,b,data):
    a_1 = transform_data(a,b,227,227,data)
    return pool5_data(a_1)

def process_data_conv5(a,b,data):
    a_1 = transform_data(a,b,227,227,data)
    return conv5_data(a_1)

def inverse_process_data(a,b,data):
    a_1 = inverse_conv5_data(data)
    return inverse_transform_data(a,b,227,227,a_1)

def rectangle_score( data, data_ ):
    a = intersection_over_union ( data, data_)
    b = distance_between_centers( data, data_)

    return ( math.e ** (-b*b) ) * a

def enumarate_neuron_pool5( data ):
    (xmin,xmax,ymin,ymax) = data
    numx = xmax - 1 + 0.5 * (11 - xmin) * xmin
    numy = ymax - 1 + 0.5 * (11 - ymin) * ymin
    return int ( 21 * numx + numy )

def enumarate_neuron_conv5( data ):
    (xmin,xmax,ymin,ymax) = data
    numx = xmax - 1 + 0.5 * (25 - xmin) * xmin
    numy = ymax - 1 + 0.5 * (25 - ymin) * ymin
    return int ( 91 * numx + numy )

def build_dictionary_pool5():
    numeration_dictionary = {}
    for xmin in range(7):
        for xmax in range(xmin+1, 7):
            for ymin in range(7):
                for ymax in range(ymin+1, 7):
                    data = (xmin,xmax,ymin,ymax)
                    numeration_dictionary[enumarate_neuron_pool5(data)] = data
    return numeration_dictionary

def build_dictionary_conv5():
    numeration_dictionary = {}
    for xmin in range(14):
        for xmax in range(xmin+1, 14):
            for ymin in range(14):
                for ymax in range(ymin+1, 14):
                    data = (xmin,xmax,ymin,ymax)
                    numeration_dictionary[enumarate_neuron_conv5(data)] = data
    return numeration_dictionary

def get_bndbox_by_id_pool5( num, a, b ):
    numeration_dictionary = build_dictionary_pool5()
    pool5_bndbox = numeration_dictionary[num]
    input_bndbox = inverse_pool5_data(pool5_bndbox)
    return inverse_transform_data(a,b,227,227,input_bndbox)

def get_bndbox_by_id_conv5( num, a, b ):
    numeration_dictionary = build_dictionary_conv5()
    pool5_bndbox = numeration_dictionary[num]
    input_bndbox = inverse_pool5_data(pool5_bndbox)
    return inverse_transform_data(a,b,227,227,input_bndbox)

def inverse_enumerate_neuron_pool5( num ):
    numeration_dictionary = build_dictionary_pool5()
    return numeration_dictionary[num]

def inverse_enumerate_neuron_conv5( num ):
    numeration_dictionary = build_dictionary_conv5()
    return numeration_dictionary[num]

def prepare_data(all_data):
    vld_data = []
    trn_data = []

    for i in range(len(all_data)):
        if i < VALIDATION_SIZE:
            vld_data.append(all_data[i])
        else:
            trn_data.append(all_data[i])

    return vld_data, trn_data


def get_next_trn_batch_pool5(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data[0]) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[0][trn_indices[i]][0] )

    batch_y = []
    for m in range(len(all_data)):
        batch_y.append(np.zeros((BATCH_SIZE,441)))
        for i in range(BATCH_SIZE):
            a = np.zeros((441))
            for data_ in all_data[m][trn_indices[i]][1]:
                (aa,bb,cc) = imread( all_data[m][trn_indices[i]][0] ).shape
                data_ = process_data_pool5(bb,aa, data_ )
                for xmin in range(7):
                    for xmax in range(xmin + 1, 7):
                        for ymin in range(7):
                            for ymax in range(ymin + 1, 7):
                                data = (xmin,xmax, ymin, ymax)
                                score = rectangle_score(data, data_ )
                                neuron = enumarate_neuron_pool5(data)
                                a[neuron] = max(a[neuron], score)

            for j in range(441):
                batch_y[m][i][j] = a[j]

    batch_z = []
    for m in range(len(all_data)):
        batch_z.append( np.zeros((BATCH_SIZE, 2)) )
        for i in range(BATCH_SIZE):
            if len(all_data[m][trn_indices[i]][1]) == 0:
                batch_z[m][i][0] = 0
                batch_z[m][i][1] = 1
            else:
                batch_z[m][i][0] = 1
                batch_z[m][i][1] = 0

    return (batch_x,batch_y,batch_z)

def get_next_trn_batch_conv5(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data[0]) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[0][trn_indices[i]][0] )

    batch_y = []
    for m in range(len(all_data)):
        batch_y.append(np.zeros((BATCH_SIZE,8281)))
        for i in range(BATCH_SIZE):
            a = np.zeros((8281))
            for data_ in all_data[m][trn_indices[i]][1]:
                (aa,bb,cc) = imread( all_data[m][trn_indices[i]][0] ).shape
                data_ = process_data_conv5(bb,aa, data_ )
                for xmin in range(14):
                    for xmax in range(xmin + 1, 14):
                        for ymin in range(14):
                            for ymax in range(ymin + 1, 14):
                                data = (xmin,xmax, ymin, ymax)
                                score = rectangle_score(data, data_ )
                                neuron = enumarate_neuron_conv5(data)
                                a[neuron] = max(a[neuron], score)

            for j in range(8281):
                batch_y[m][i][j] = a[j]

    batch_z = []
    for m in range(len(all_data)):
        batch_z.append( np.zeros((BATCH_SIZE, 2)) )
        for i in range(BATCH_SIZE):
            if len(all_data[m][trn_indices[i]][1]) == 0:
                batch_z[m][i][0] = 0
                batch_z[m][i][1] = 1
            else:
                batch_z[m][i][0] = 1
                batch_z[m][i][1] = 0

    return (batch_x,batch_y,batch_z)

def get_next_trn_batch_conv5_convolution(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data[0]) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[0][trn_indices[i]][0] )

    batch_y = [[ [] for i in range(14)] for j in range(14)]
    for h in range(1,14):
        for w in range(1,14):
            for m in range(len(all_data)):
                batch_y[h][w].append(np.zeros((BATCH_SIZE,14-h,14-w)))
                for i in range(BATCH_SIZE):
                    a = np.zeros((14-h,14-w))
                    for data_ in all_data[m][trn_indices[i]][1]:
                        (aa,bb,cc) = imread( all_data[m][trn_indices[i]][0] ).shape
                        data_ = process_data_conv5(bb,aa, data_ )
                        for x in range(14 - h):
                            for y in range(14 - w):
                                data = (y,y+w, x, x+h)
                                score = rectangle_score(data, data_ )
                                a[x][y] = max(a[x][y], score)

                    for x in range(14 - h):
                        for y in range(14 - w):
                            batch_y[h][w][m][i][x][y] = a[x][y]

    batch_z = []
    for m in range(len(all_data)):
        batch_z.append( np.zeros((BATCH_SIZE, 2)) )
        for i in range(BATCH_SIZE):
            if len(all_data[m][trn_indices[i]][1]) == 0:
                batch_z[m][i][0] = 0
                batch_z[m][i][1] = 1
            else:
                batch_z[m][i][0] = 1
                batch_z[m][i][1] = 0

    return (batch_x,batch_y,batch_z)


def get_next_trn_batch_face_conv5_convolution(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_y = [[ [] for i in range(14)] for j in range(14)]
    for h in range(1,14):
        for w in range(1,14):
            batch_y[h][w] = np.zeros((BATCH_SIZE,14-h,14-w))
            for i in range(BATCH_SIZE):
                a = np.zeros((14-h,14-w))
                for data_ in all_data[trn_indices[i]][1]:
                    (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
                    data_ = process_data_conv5(bb,aa, data_ )
                    for x in range(14 - h):
                        for y in range(14 - w):
                            data = (y,y+w, x, x+h)
                            score = rectangle_score(data, data_ )
                            a[x][y] = max(a[x][y], score)

                for x in range(14 - h):
                    for y in range(14 - w):
                        batch_y[h][w][i][x][y] = a[x][y]

    return (batch_x,batch_y)

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


def get_next_trn_batch_and_paths_face_conv5_convolution(all_data):
    trn_indices = []
    for i in range(BATCH_SIZE):
        trn_indices.append(random.randint(0,len(all_data) - 1))

    path = all_data[trn_indices[0]][0]

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(BATCH_SIZE):
        batch_x[i,:,:,:,] = image_preprocess( all_data[trn_indices[i]][0] )

    batch_y = [[ [] for i in range(14)] for j in range(14)]
    for h in range(1,14):
        for w in range(1,14):
            batch_y[h][w] = np.zeros((BATCH_SIZE,14-h,14-w))
            for i in range(BATCH_SIZE):
                a = np.zeros((14-h,14-w))
                for data_ in all_data[trn_indices[i]][1]:
                    (aa,bb,cc) = imread( all_data[trn_indices[i]][0] ).shape
                    data_ = process_data_conv5(bb,aa, data_ )
                    for x in range(14 - h):
                        for y in range(14 - w):
                            data = (y,y+w, x, x+h)
                            score = rectangle_score(data, data_ )
                            a[x][y] = max(a[x][y], score)

                for x in range(14 - h):
                    for y in range(14 - w):
                        batch_y[h][w][i][x][y] = a[x][y]

    return (batch_x,batch_y,path)


def detection_accuracy(object_or_not, truth, answer):
    correct = 0
    num_object = 0
    for i in range(BATCH_SIZE):
        if object_or_not[i][0] == 1:
            num_object += 1
            res = np.argmax( truth[i] )
            ans = answer[i].argsort()[-10:][::-1]

            for j in range(5):
                if ans[j] == res:
                    correct += 1

    return correct / float(num_object)


def face_detection_accuracy( truth, result ):
    correct = 0
    for i in range(BATCH_SIZE):

        answer_list = []
        for h in range(1,14):
            for w in range(1,14):
                for k in range(14-h):
                    for l in range(14-w):
                        confidence = truth[h][w][i][k][l]
                        data = (l,l+w,k,k+h)
                        answer_list.append( (confidence, data) )

        answer_list.sort(key=lambda tup: tup[0])
        truth_top = []
        for m in range(5):
            truth_top.append( answer_list[-m-1][1] )

        answer_list_1 = []
        for h in range(1,14):
            for w in range(1,14):
                for k in range(14-h):
                    for l in range(14-w):
                        confidence = result[h][w][i][k][l]
                        data = (l,l+w,k,k+h)
                        answer_list_1.append( (confidence, data) )

        answer_list_1.sort(key=lambda tup: tup[0])
        result_top = []
        for m in range(5):
            result_top.append( answer_list_1[-m-1][1] )

        checker = False
        for bndbox_1 in truth_top:
            for bndbox_2 in result_top:
                if bndbox_1 == bndbox_2:
                    checker = True

        if intersection_over_union(truth_top[0], result_top[0]) > 0.2:
        #if checker:
            correct += 1

    return float(correct) / BATCH_SIZE

def draw_detection(path, result):

    answer_list = []
    for h in range(1,14):
        for w in range(1,14):
            for k in range(14-h):
                for l in range(14-w):
                    confidence = result[h][w][0][k][l]
                    data = (l,l+w,k,k+h)
                    answer_list.append( (confidence, data) )

    answer_list.sort(key=lambda tup: tup[0])
    result_top = []
    for m in range(1):
        result_top.append( answer_list[-m-1][1] )

    data = result_top[0]
    (a,b,c) = imread(path).shape
    data = inverse_process_data(b,a,data)
    show_detection(path, data)

def get_detection_list_from_ids(detections, a, b):
    detection_list = []
    for i in range(len(detections)):
        (xmin,xmax,ymin,ymax) = get_bndbox_by_id(i,a,b)
        width = xmax - xmin
        height = ymax - ymin
        if detections[i] > 0.05:
            detection_list.append( (xmin, ymin, detections[i], width, height) )
    return detection_list


def process_and_write_detections( g, name, detection_list ):
    new_detection_list = detection.nms(detection_list)
    for det in new_detection_list:
        bndbox = (det[0] + 1, det[1] + 1, det[0] + det[3] + 1, det[1] + det[4] + 1)
        res = name + " " + str(det[2]) + " " + str(bndbox[0]) + " " + str(bndbox[1]) + \
                " " + str(bndbox[2]) + " " + str(bndbox[3]) + "\n"
        g.write(res)

def just_write_max_detection( g, name, a, b, detection_list):
    bndbox = get_bndbox_by_id_pool5( np.argmax(detection_list), a, b)
    pool5_bndbox = inverse_enumerate_neuron_pool5(np.argmax(detection_list))
    confidence = detection_list[np.argmax(detection_list)]
    res = name + " " + str(confidence) + " " + str(bndbox[0] + 1) + " " + str(bndbox[2] + 1) + \
            " " + str(bndbox[1] + 1) + " " + str(bndbox[3] + 1) + "    " \
            + str(pool5_bndbox[0]) + " " + str(pool5_bndbox[1]) + " " + str(pool5_bndbox[2]) + \
            " " + str(pool5_bndbox[3]) + "\n"
    g.write(res)

def show_detection(path, bndbox):
    global CTR #worst idea ever
    xmin,xmax,ymin,ymax = bndbox
    xmin = int(xmin + 1); xmax = int(xmax - 1);
    ymin = int(ymin + 1); ymax = int(ymax - 1);

    img = mpimg.imread(path)

    for i in range(xmin, xmax):
        img[ymin][i][0] = 255
        img[ymin][i][1] = 0
        img[ymin][i][2] = 0

    for i in range(xmin, xmax):
        img[ymax][i][0] = 255
        img[ymax][i][1] = 0
        img[ymax][i][2] = 0

    for i in range(ymin, ymax):
        img[i][xmin][0] = 255
        img[i][xmin][1] = 0
        img[i][xmin][2] = 0

    for i in range(ymin, ymax):
        img[i][xmax][0] = 255
        img[i][xmax][1] = 0
        img[i][xmax][2] = 0

    a = Image.fromarray(img)
    a.save(root + "/Desktop/"+ str(CTR) + ".jpg" )
    CTR += 1
