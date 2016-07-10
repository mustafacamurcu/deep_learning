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
sys.path.append(root + 'VGG_Face/')
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from VGG_Face import VGG_Face

train_jpg, train_txt = utils.MTFL_directories("train")
test_jpg, test_txt = utils.MTFL_directories("test")
train_data = utils.import_MTFL_data(train_jpg,train_txt)
test_data  = utils.import_MTFL_data(test_jpg,test_txt)

x = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,224,224,3])
net = VGG_Face({'data' : x}, trainable = True)

u = VGG_graph.VGG_face_scratch_point_detection_net_GMM(net)
loss = u[0]; mean_x = u[1]; mean_y = u[2]; x_ = u[3]; y_ = u[4]; loss2 = u[5];
structural_gradient = u[6]; visual_gradient = u[7];

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
ITERATIONS = 1000000

f = open(root + "Experiments/Results/VGG_face_scratch_train_gmm_log_conv5_5.txt", "w")
g = open(root + "Experiments/Results/VGG_face_scratch_test_gmm_log_conv5_5.txt", "w")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    while ITERATIONS > 0:
        batch_x,batch_point_x,batch_point_y = VGG_utils.get_next_trn_batch_scratch_face(train_data)
        _,error,mx,my,l2,sg,vg = sess.run( [train_step,loss,mean_x,mean_y,loss2, structural_gradient, visual_gradient],
                                        feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y } )

        ITERATIONS -= 1
        sys.stdout.write('\r\x1b[K')
        sys.stdout.write("Train Error: %lf loss2: %lf Remaining Iterations: %d sg: %lf vg: %lf" %(error,l2, ITERATIONS,sg[0,0],vg[0,0]))
        sys.stdout.flush()

        if ITERATIONS % 10 == 0:
            sys.stdout.write('\n')

        if ITERATIONS % 30 == 0:
            f.write(str(error) + "\n")
            f.flush()
            error = 0; l2 = 0;
            for i in range(10):
                batch_x,batch_point_x,batch_point_y = VGG_utils.get_next_trn_batch_scratch_face(test_data)
                a = sess.run( [loss,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y} )
                error += a[0]; l2 += a[1]
            error /= 10.; l2 /= 10.;
            sys.stdout.write("Validation Error: %lf loss2: %lf Remaining Iterations: %d" %(error, l2, ITERATIONS))
            sys.stdout.write('\n')
            g.write(str(error) + "\n")
            g.flush()

        if ITERATIONS % 200 == 0:
            saver.save(sess, root + 'Experiments/Models/VGG_face_scratch_model_conv5_5_gmm')
f.close()
g.close()
