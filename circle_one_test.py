import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
import utils
import graph
#ronaldo
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from mynet import AlexNet
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

x = tf.placeholder(tf.float32, shape = [1,227,227,3])
net = AlexNet({'data' : x}, trainable = False)

y = graph.face_test_net_conv5_convolution(net)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'caltech-model-trainable-convolution-multiply')
    correct = 0
    directory = utils.root + "/Desktop/CaltechData/Images/pic00008.jpg"
    (a,b,c) = utils.imread(directory).shape
    batch_x = utils.image_preprocess(directory)
    batch_x.resize((1,227,227,3))
    detection = [[0 for j in range(14)] for i in range(14)]

    for h in range(1,14):
        for w in range(1,14):
            detection[h][w] = sess.run( y[h][w], feed_dict = {x :  batch_x } )

    data = (0,0,0,0)
    confidence = 0

    answer_list_1 = []
    for h in range(1,14):
        for w in range(1,14):
            for k in range(14-h):
                for l in range(14-w):
                    confidence = detection[h][w][0][k][l]
                    data = (l,l+w,k,k+h)
                    answer_list_1.append( (confidence, data) )

    answer_list_1.sort(key=lambda tup: tup[0])
    result_top = []
    for m in range(5):
        print answer_list_1[-m-1][0]
        result_top.append( answer_list_1[-m-1][1] )

    for data in result_top:
        data = utils.inverse_process_data(b,a,data)
        utils.show_detection(directory, data)
