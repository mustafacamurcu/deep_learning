import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import math
import time
import random
import matplotlib.image as mpimg
import glob
import utils
import VGG_utils
import VGG_graph
root = '/data/vision/torralba/health-habits/other/enes/'
sys.path.append(root + 'VGG_Classic/')
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from VGG_Classic import VGG_Classic

root1 = '/afs/csail.mit.edu/u/k/kocabey/Desktop/'
root2 = '/data/vision/torralba/health-habits/other/enes/'

test_file = root2 + 'HumanData/Train/'

ground_truth = np.zeros((1000,14,2))
txt = glob.glob(test_file + 'annotations/*.txt')
jpg = glob.glob(test_file + 'images/*.jpg')

for i in range(len(txt)):
    f = open(txt[i])
    data = f.readlines()
    for j in range(14):
        ground_truth[i][j] = data[j].split()[:2]

def dist(x1, x2, y1 ,y2):
    return math.sqrt( (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) )

def check_part(joint1, joint2, img_id, img_path, sess):
    global ground_truth
    j1 = ground_truth[img_id][joint1]
    j2 = ground_truth[img_id][joint2]

    d = dist(j1[0], j2[0], j1[1], j2[1]) #half the length of the part

    batch_x = np.zeros((1,224,224,3))

    batch_x[0,:,:,:,] = VGG_utils.image_preprocess(img_path)

    mx,my = sess.run([mean_x,mean_y], feed_dict = {x: batch_x})

    img = mpimg.imread(jpg[i])

    mx = mx * img.shape[0] / 10.
    my = my * img.shape[1] / 10.

    if dist(mx[0][joint1], j1[0], my[0][joint1], j1[1]) < d:
        if dist(mx[0][joint2], j2[0], my[0][joint2], j2[1]) < d:
            return True
    else:
        return False

def accuracy(sess, jpg):
    r_lower_leg = 0
    for i in range(len(jpg)):
        if check_part(0,1,i,jpg[i],sess):
            r_lower_leg += 1

    print r_lower_leg / float(len(jpg))

    r_upper_leg = 0
    for i in range(len(jpg)):
        if check_part(1,2,i,jpg[i],sess):
            r_upper_leg += 1

    print r_upper_leg / float(len(jpg))

    l_upper_leg = 0
    for i in range(len(jpg)):
        if check_part(3,4,i,jpg[i],sess):
            l_upper_leg += 1

    print l_upper_leg / float(len(jpg))

    l_lower_leg = 0
    for i in range(len(jpg)):
        if check_part(4,5,i,jpg[i],sess):
            l_lower_leg += 1

    print l_lower_leg / float(len(jpg))

    r_fore_arm = 0
    for i in range(len(jpg)):
        if check_part(6,7,i,jpg[i],sess):
            r_fore_arm += 1

    print r_fore_arm / float(len(jpg))

    r_upper_arm = 0
    for i in range(len(jpg)):
        if check_part(7,8,i,jpg[i],sess):
            r_upper_arm += 1

    print r_lower_leg / float(len(jpg))

    l_upper_arm = 0
    for i in range(len(jpg)):
        if check_part(9,10,i,jpg[i],sess):
            l_upper_arm += 1

    print l_upper_arm / float(len(jpg))

    l_fore_arm = 0
    for i in range(len(jpg)):
        if check_part(10,11,i,jpg[i],sess):
            l_fore_arm += 1

    print l_fore_arm / float(len(jpg))

    head = 0
    for i in range(len(jpg)):
        if check_part(12,13,i,jpg[i],sess):
            head += 1

    print head / float(len(jpg))

x = tf.placeholder(tf.float32, shape = [1,224,224,3])
net = VGG_Classic({'data' : x}, trainable = True)

W = tf.Variable(tf.random_uniform([5,5,512,14],-1e-2,1e-2))
b = tf.Variable(tf.random_uniform([14],-1e-2,1e-2))

conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5_2'], W, [1,1,1,1], 'VALID'), b )
conv = tf.nn.relu(conv)

total = tf.reduce_sum(conv, [1,2], True)
total = tf.clip_by_value(total,1e-9,1000000000)
conv /= total

mean_x, mean_y = 0,0

for i in range(10):
    for j in range(10):
        mean_x += conv[:,i,j,:] * (i + 0.5)
        mean_y += conv[:,i,j,:] * (j + 0.5)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, root2 + 'Experiments/Models/VGG_human_model_conv5_2')

accuracy(sess, jpg)
