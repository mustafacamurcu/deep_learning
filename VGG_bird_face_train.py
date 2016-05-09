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

train_jpg, train_txt = utils.bird_random_slice_directories("train")
test_jpg, test_txt = utils.bird_random_slice_directories("test")
train_data = utils.import_bird_point_data(train_jpg,train_txt)
test_data  = utils.import_bird_point_data(test_jpg,test_txt)

x = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,224,224,3])
net = VGG_Face({'data' : x}, trainable = True)

u = VGG_graph.VGG_bird_point_detection_net(net)
loss = u[0]; mean_x = u[1]; mean_y = u[2]; x_ = u[3]; y_ = u[4]; z_ = u[5]; loss2 = u[6];

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

saver = tf.train.Saver()
ITERATIONS = 1000000

f = open(root + "Experiments/Results/VGG_bird_face_train_log_conv5_5_new.txt", "w")
g = open(root + "Experiments/Results/VGG_bird_face_test_log_conv5_5_new.txt", "w")

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    net.load(root + "VGG_Face/VGG_Face.npy", sess)
    print "VGG Network has been successfully uploaded!"
    #saver.restore(sess,root + "Experiments/Models/VGG_bird_model_conv5_5_new")
    while ITERATIONS > 0:
        batch_x,batch_point_x,batch_point_y,batch_existence = VGG_utils.get_next_trn_batch_bird(train_data)
        _, error,mx,my,l2 = sess.run( [train_step,loss,mean_x,mean_y,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y, z_: batch_existence } )

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
                batch_x,batch_point_x,batch_point_y,batch_existence = VGG_utils.get_next_trn_batch_bird(test_data)
                a = sess.run( [loss,loss2], feed_dict = {x: batch_x, x_: batch_point_x, y_: batch_point_y, z_: batch_existence } )
                error += a[0]; l2 += a[1]
            error /= 10.; l2 /= 10.;
            sys.stdout.write("Validation Error: %lf loss2: %lf Remaining Iterations: %d" %(error, l2, ITERATIONS))
            sys.stdout.write('\n')
            g.write(str(error) + "\n")
            g.flush()

        if ITERATIONS % 200 == 0:
            saver.save(sess, root + 'Experiments/Models/VGG_bird_face_model_conv5_5_new')
f.close()
g.close()
