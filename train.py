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

all_data = []; vld_data = []; trn_data = [];

for name in utils.SAMPLE_VOC_NAMES:
    all_data.append( utils.import_data(name) )

x = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,227,227,3])
net = AlexNet({'data' : x}, trainable = True)

y_ = []; z_ = []; y = []; z = []; losses = []; accuracies = [];

for i in range(len(all_data)):
    print i
    u = graph.localization_net_conv5_convolution(net)
    y_.append(u[0])
    z_.append(u[1])
    y.append(u[2])
    z.append(u[3])
    losses.append(u[4])
    accuracies.append(u[5])

total_loss = 0
for loss in losses:
    total_loss += loss

train_step = tf.train.AdamOptimizer(1e-4).minimize(total_loss)

saver = tf.train.Saver()
ITERATIONS = 200

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/mynet.npy', sess)
    a = True
    while ITERATIONS > 0:
        batch_x,batch_y,batch_z = utils.get_next_trn_batch_conv5_convolution(all_data)
        f_dict = {}
        f_dict[x] = batch_x
        for i in range(len(all_data)):
            for h in range(1,14):
                for w in range(1,14):
                    f_dict[y_[i][h][w]] = batch_y[h][w][i]
            #f_dict[y_[i]] = batch_y[i]
            f_dict[z_[i]] = batch_z[i]

        loss = sess.run(losses, feed_dict = f_dict )
        sess.run( train_step, feed_dict = f_dict )
        print loss[0]

        ITERATIONS -= 1
        print "Remaining Iterations: ", ITERATIONS

    saver.save(sess, 'my-model-trainable-convolution-IoUscore', global_step=0)
