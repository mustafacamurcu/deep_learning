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

all_data,all_name_data = utils.import_test_data()

x = tf.placeholder(tf.float32, shape = [1,227,227,3])
net = AlexNet({'data' : x}, trainable = False)

y = []; z = [];

for i in range(len(utils.SAMPLE_VOC_NAMES)):
    print i
    u = graph.test_net_conv5_convolution(net)
    y.append(u[0])
    z.append(u[1])

saver = tf.train.Saver()

f = []; g = [];

for i in range(len(utils.SAMPLE_VOC_NAMES)):
    filename = "comp1_cls_test_" + utils.SAMPLE_VOC_NAMES[i] + ".txt"
    f.append(open(filename,"w"))
    filename = "comp3_det_test_" + utils.SAMPLE_VOC_NAMES[i] + ".txt"
    g.append(open(filename,"w"))

with tf.Session() as sess:
    net.load('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/mynet.npy', sess)
    saver.restore(sess, 'my-model-trainable-conv5-convolution-0')
    u = 0
    for m in range(len(all_data)):
        directory = all_data[m]
        name = all_name_data[m]
        (a,b,c) = utils.imread(directory).shape
        batch_x = utils.image_preprocess(directory)
        batch_x.resize((1,227,227,3))
        detection = [[0 for j in range(14)] for i in range(14)]

        for h in range(1,14):
            for w in range(1,14):
                detection[h][w] = sess.run( y[0][h][w], feed_dict = {x :  batch_x } )

        #detection = sess.run( y, feed_dict = {x : batch_x } )
        classification = sess.run( z, feed_dict = {x : batch_x } )

        for i in range(len(utils.SAMPLE_VOC_NAMES)):
            res = name + " " + str(classification[i][0][0]) + "\n"
            f[i].write(res)
            #if you want nms activate this part
            #detection_list = utils.get_detection_list_from_ids(detection[i][0], a, b)
            #utils.process_and_write_detections(g[i], name, detection_list)
            #utils.just_write_max_detection(g[i],name, a,b,detection[i][0])
            for h in range(1,14):
                for w in range(1,14):
                    for k in range(14-h):
                        for l in range(14-w):
                            if detection[h][w][0][k][l] > 0.3:
                                g[i].write(name + " " + str(detection[h][w][0][k][l]) + " " +\
                                       str(l) + " " + str(k) + " " + str(l + w) + " " + str(k + h) + "\n")

            #print '\n\n\n'
        u += 1
        print u

    for i in range(len(utils.SAMPLE_VOC_NAMES)):
        f[i].close()
        g[i].close()
