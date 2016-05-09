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

f = open(utils.root + "/Desktop/log.txt", "w")

all_data = utils.import_caltech_data()

x = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,227,227,3])
net = AlexNet({'data' : x}, trainable = False)

u = graph.face_localization_net_conv5_convolution(net)
y_ = u[0]; y = u[1]; loss = u[2]

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
ITERATIONS = 100000

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/mynet.npy', sess)
    a = True
    while ITERATIONS > 0:
        batch_x,batch_y,path = utils.get_next_trn_batch_and_paths_face_conv5_convolution(all_data)
        f_dict = {}
        f_dict[x] = batch_x
        for h in range(1,14):
            for w in range(1,14):
                f_dict[y_[h][w]] = batch_y[h][w]

        u = []
        result = [[0 for j in range(14)] for i in range(14)]
        for h in range(1,14):
            for w in range(1,14):
                u.append(y[h][w])

        output = sess.run(u, feed_dict = f_dict )

        index = 0
        for h in range(1,14):
            for w in range(1,14):
                result[h][w] = output[index]
                index += 1

        acc = utils.face_detection_accuracy(batch_y, result)
        utils.draw_detection(path, result)
        _, error = sess.run( [train_step,loss], feed_dict = f_dict )
        ITERATIONS -= 1
        f.write( str(acc) + " " + str(error) )
        sys.stdout.write('\r\x1b[K')
        sys.stdout.write("Accuracy: %lf Error: %lf Remaining Iterations: %d" %(acc,error,ITERATIONS))
        sys.stdout.flush()

        if ITERATIONS % 50 == 0:
            saver.save(sess, 'caltech-model-trainable-convolution-multiply')
