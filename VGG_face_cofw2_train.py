import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
import utils
import VGG_utils
import VGG_graph
root = '/data/vision/torralba/health-habits/other/enes/'
sys.path.append(root + 'VGG_Classic/')
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from VGG_Classic import VGG_Classic

train_jpg, train_txt = utils.COFW2_directories("train")
test_jpg, test_txt = utils.COFW2_directories("test")
train_data = utils.import_COFW_data(train_jpg,train_txt)
test_data  = utils.import_COFW_data(test_jpg,test_txt)

x = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,224,224,3])
net = VGG_Classic({'data' : x}, trainable = True)

u = VGG_graph.VGG_face_29_point_detection_net(net)
loss = u[0]; mean_x = u[1]; mean_y = u[2]; x_ = u[3]; y_ = u[4]; loss2 = u[5];

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
ITERATIONS = 1000000

f = open(root + "Experiments/Results/VGG_face_COFW2_train_log_conv5_5.txt", "w")
g = open(root + "Experiments/Results/VGG_face_COFW2_test_log_conv5_5.txt", "w")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print "VGG Network has been successfully uploaded!"
    while ITERATIONS > 0:
        batch_x,batch_point_x,batch_point_y = VGG_utils.get_next_trn_batch_face_COFW2(train_data)
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
                batch_x,batch_point_x,batch_point_y = VGG_utils.get_next_trn_batch_face_COFW2(test_data)
                a = sess.run( [loss,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y} )
                error += a[0]; l2 += a[1]
            error /= 10.; l2 /= 10.;
            sys.stdout.write("Validation Error: %lf loss2: %lf Remaining Iterations: %d" %(error, l2, ITERATIONS))
            sys.stdout.write('\n')
            g.write(str(error) + "\n")
            g.flush()

        if ITERATIONS % 200 == 0:
            saver.save(sess, root + 'Experiments/Models/VGG_face_COFW2_model_conv5_5')
f.close()
g.close()
