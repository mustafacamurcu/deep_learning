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

train_jpg, train_txt = utils.caltech_random_slice_directories("train")
test_jpg, test_txt = utils.caltech_random_slice_directories("test")
train_data = utils.import_caltech_point_data(train_jpg,train_txt)
test_data  = utils.import_caltech_point_data(test_jpg,test_txt)

x = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,227,227,3])
net = AlexNet({'data' : x}, trainable = True)

u = graph.face_visual_point_detection_net(net)
loss = u[0]; mean_x = u[1]; mean_y = u[2]; x_ = u[3]; y_ = u[4]; loss2 = u[5];

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
ITERATIONS = 1000000

f = open("train_variance_alog.txt", "w")
g = open("test_variance_alog.txt", "w")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/mynet.npy', sess)
    while ITERATIONS > 0:
        batch_x,batch_point_x,batch_point_y = utils.get_next_trn_batch_face_point_conv5_convolution(train_data)
        _, error,mx,my,l2 = sess.run( [train_step,loss,mean_x,mean_y,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y } )

        ITERATIONS -= 1
        sys.stdout.write('\r\x1b[K')
        sys.stdout.write("Train Error: %lf loss2: %lf Remaining Iterations: %d" %(error,l2, ITERATIONS))
        sys.stdout.flush()

        if ITERATIONS % 10 == 0:
            sys.stdout.write('\n')

        if ITERATIONS % 30 == 0:
            f.write(str(error) + "\n")
            f.flush()
            error = 0; l2 = 0;
            for i in range(10):
                batch_x,batch_point_x,batch_point_y = utils.get_next_trn_batch_face_point_conv5_convolution(test_data)
                a = sess.run( [loss,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y } )
                error += a[0]; l2 += a[1]
            error /= 10.; l2 /= 10.;
            sys.stdout.write("Validation Error: %lf loss2: %lf Remaining Iterations: %d" %(error, l2, ITERATIONS))
            sys.stdout.write('\n')
            g.write(str(error) + "\n")
            g.flush()

        if ITERATIONS % 100 == 0:
            saver.save(sess, 'caltech-point-avariance-model')
f.close()
g.close()
