import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random

sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from mynet import AlexNet
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from os import listdir
import xml.etree.ElementTree as ET

xml_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit3/VOC2010/Annotations/')
jpg_directories = listdir('/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit3/VOC2010/JPEGImages/')

BATCH_SIZE = 256
VALIDATION_SIZE = 256

all_data = []
for i in range( 1, len(xml_directories)):
    xml = xml_directories[i]
    directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit3/VOC2010/Annotations/' + xml
    tree = ET.parse(directory)
    root = tree.getroot()

    data = []

    for obj in root.findall( 'object' ):
        if obj.find('name').text == 'cat':
            bndbox = obj.find( 'bndbox' )
            data.append( ( int(bndbox.find('xmin').text), int(bndbox.find('xmax').text),
                          int(bndbox.find('ymin').text), int(bndbox.find('ymax').text) ) )

    directory = '/afs/csail.mit.edu/u/k/kocabey/Desktop/VOCdevkit3/VOC2010/JPEGImages/' + jpg_directories[i]
    all_data.append( (directory , data) )

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

RATIO = 227 / float(6)

def pool5_data( data ):
    xmin = math.floor ( data[0] / RATIO )
    xmax = math.ceil  ( data[1] / RATIO )
    ymin = math.floor ( data[2] / RATIO )
    ymax = math.ceil  ( data[3] / RATIO )
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

def process_data(a,b,data):
    a_1 = transform_data(a,b,227,227,data)
    return pool5_data(a_1)

def rectangle_score( data, data_ ):
    a = intersection_over_union ( data, data_)
    b = distance_between_centers( data, data_)

    return ( ( math.e ** (-b*b) ) + a ) / 2

def enumarate_neuron( data ):
    (xmin,xmax,ymin,ymax) = data
    numx = xmax - 1 + 0.5 * (11 - xmin) * xmin
    numy = ymax - 1 + 0.5 * (11 - ymin) * ymin
    return int ( 21 * numx + numy )

vld_data = []
trn_data = []

for i in range( len(all_data) ):
    if i < VALIDATION_SIZE:
        vld_data.append(all_data[i])
    else:
        trn_data.append(all_data[i])

trn_index = 0

def get_next_trn_batch():
    global trn_index

    if trn_index + BATCH_SIZE > len(trn_data):
        random.shuffle( trn_data )
        trn_index = 0

    start = trn_index
    end = trn_index + BATCH_SIZE
    trn_index += BATCH_SIZE

    batch_x = np.zeros((BATCH_SIZE,227,227,3))
    for i in range(start, end):
        batch_x[i - start,:,:,:,] = image_preprocess( trn_data[i][0] )

    batch_y = np.zeros((BATCH_SIZE,441))
    for i in range(start, end):
        a = np.zeros((441))
        for data_ in trn_data[i][1]:
            (aa,bb,cc) = imread( trn_data[i][0] ).shape
            data_ = process_data(bb,aa, data_ )
            for xmin in range(7):
                for xmax in range(xmin + 1, 7):
                    for ymin in range(7):
                        for ymax in range(ymin + 1, 7):
                            data = (xmin,xmax, ymin, ymax)
                            score = rectangle_score(data, data_ )
                            neuron = enumarate_neuron(data)
                            a[neuron] = max(a[neuron], score)

        for j in range(441):
            batch_y[i- trn_index][j] = a[j]


    batch_z = np.zeros((BATCH_SIZE, 1))
    for i in range(start,end):
        if len(trn_data[i][1]) == 0:
            batch_z[i - trn_index][0] = 0
        else:
            batch_z[i - trn_index][0] = 1

    return (batch_x, batch_y, batch_z)


x = tf.placeholder(tf.float32, shape = [BATCH_SIZE,227,227,3])
y_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE,441])
z_ = tf.placeholder(tf.float32, shape = [BATCH_SIZE,1])

net = AlexNet({'data' : x}, trainable = False)

W = tf.Variable(tf.random_uniform([6,6,256,441],-0.1,0.1))
b = tf.Variable(tf.random_uniform([441],-0.1,0.1))
W_cls = tf.Variable(tf.random_uniform([4096,1],-0.1,0.1))
b_cls = tf.Variable(tf.random_uniform([1],-0.1,0.1))

fc = tf.nn.conv2d(net.layers['pool5'], W, [1,1,1,1], 'VALID')
res = tf.nn.bias_add(fc,b)
res = tf.reshape(res, [BATCH_SIZE,441])
y = tf.sigmoid(res)

fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
z = tf.sigmoid(fc_cls)

loss_detection = tf.reduce_mean( tf.square(y - y_ ))
loss_classification = tf.reduce_mean( tf.square(z - z_) )

loss = loss_detection + 10 * loss_classification
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/mynet.npy', sess)
    while True:
        batch_x, batch_y, batch_z = get_next_trn_batch()
        error_detection, error_classification, _ = sess.run([loss_detection, loss_classification, train_step],
                         feed_dict = {x : batch_x, y_: batch_y, z_ : batch_z})
        print error_detection, error_classification
