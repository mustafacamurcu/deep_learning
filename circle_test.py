import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
import utils
import graph

sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from mynet import AlexNet
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

all_data = utils.import_circle_test_data()

x = tf.placeholder(tf.float32, shape = [1,227,227,3])
net = AlexNet({'data' : x}, trainable = False)

y = graph.face_test_net_conv5_convolution(net)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'circle-model-trainable-convolution-multiply-0')
    correct = 0
    for m in range(len(all_data)):
        directory = all_data[m][0]
        (a,b,c) = utils.imread(directory).shape
        batch_x = utils.image_preprocess(directory)
        batch_x.resize((1,227,227,3))
        detection = [[0 for j in range(14)] for i in range(14)]

        for h in range(1,14):
            for w in range(1,14):
                detection[h][w] = sess.run( y[h][w], feed_dict = {x :  batch_x } )

        data_ = all_data[m][1][0]
        data = (0,0,0,0)
        confidence = 0

        for h in range(1,14):
            for w in range(1,14):
                for k in range(14-h):
                    for l in range(14-w):
                        if detection[h][w][0][k][l] > confidence:
                            confidence = detection[h][w][0][k][l]
                            data = (l,l+w,k,k+h)

        data = utils.inverse_process_data(b,a,data)

        if utils.intersection_over_union(data_, data) > 0.5:
            correct += 1
            print "correct", m, data_, data
        else:
            print "wrong", m, data_, data

    print "Accuracy: ", float(correct) / len(all_data)
