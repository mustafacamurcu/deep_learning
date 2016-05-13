# Evaluation Criteria:
# A guess is considered [Correct Guess]
# [1] If part is invisible: guess must say invisible
# [2] If part is visible: guess must say part is visible and
# guess has to be within 1.5 Standard Deviation of Amazon MTurk Worker clicks
# [Details] If ground truth file says a part is visible and MTurk Worker says part is invisible
# than MTurk Worker will be ignored in Standard Deviation calculation
# We use biased definition of Standard Deviation [Division by N]

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
sys.path.append(root + 'VGG_Bird/')
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from VGG_Bird import BIRDS_VGG_ILSVRC_16_layers

root1 = '/afs/csail.mit.edu/u/k/kocabey/Desktop/'
root2 = '/data/vision/torralba/health-habits/other/enes/'

file_name_ground_truth = root1 + 'BirdData/part_locs.txt'
file_name_MTurk = root1 + 'BirdData/part_click_locs.txt'

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def import_ground_truth(file_name):
    ground_truth = np.zeros((11789,16,3))
    f = open(file_name)
    data = f.readlines()
    for i in range(len(data)):
        image_id, part_id, x,y,z = data[i].split()[0:5]
        x = int(float(x)); y = int(float(y)); z = int(float(z))
        image_id = int(image_id)
        part_id = int(part_id)
        ground_truth[image_id][part_id] = (x,y,z)

    return ground_truth

ground_truth = import_ground_truth(file_name_ground_truth)

def get_MTurk_data(file_name):
    f = open(file_name)
    data  = f.readlines()
    f.close()
    MTurk_data = [ [ [] for j in range(16) ] for i in range(15000) ]
    for i in range(len(data)):
        datum = data[i].split()
        image_id = int(datum[0])
        part_id = int(datum[1])
        x = int(float(datum[2]))
        y = int(float(datum[3]))
        if x != 0 or y != 0:
            MTurk_data[image_id][part_id].append((x,y))

    return MTurk_data

data = get_MTurk_data(file_name_MTurk)

def mean_stddev_part(image_id, part_id):
    global data
    datum = data[image_id][part_id]
    if len(datum) == 0:
        return (0.,0.),0.

    total = [0,0]
    for i in range(len(datum)):
        total[0] += datum[i][0]
        total[1] += datum[i][1]

    mean = [0,0]

    mean[0] = total[0] / float(len(datum))
    mean[1] = total[1] / float(len(datum))

    ret = 0

    for i in range(len(datum)):
        ret += (datum[i][0] - mean[0])**2
        ret += (datum[i][1] - mean[1])**2

    ret /= float(len(datum))
    return mean, math.sqrt(ret)

def checker(image_id, part_id, guess, visibility):
    if visibility == 0:
        if guess[2] == 0:
            return True
        if guess[2] == 1:
            return True

    if visibility == 1:
        if guess[2] == 1:
            mean, stddev = mean_stddev_part(image_id, part_id)
            distance = math.sqrt((mean[0] - guess[0])**2 + (mean[1] - guess[1])**2)
            if distance <= 1.5 * stddev:
                return True
            if distance > 1.5 * stddev:
                return False
        if guess[2] == 0:
            return True

def model_accuracy(sess, part_id):
    global ground_truth
    jpg = glob.glob( root1 + 'BirdData/RealTest/*.jpg')
    correct = 0; total = 0; visibility_correct = 0;
    for i in range(100):
        image_id = int("".join(jpg[i].split(".jpg")).split("Test/")[1]) + 1
        batch_x = np.zeros((1,224,224,3))
        batch_x[0,:,:,:,] = VGG_utils.image_preprocess(jpg[i])
        mx,my,f_c = sess.run([mean_x,mean_y,fc], feed_dict = {x: batch_x})


        visibility = ground_truth[image_id][part_id][2]

        img = mpimg.imread(jpg[i])

        mx = mx * img.shape[0] / 20.
        my = my * img.shape[1] / 20.

        for s in range(15):
            mx[0][s] = int(mx[0][s])
            my[0][s] = int(my[0][s])
            if sigmoid(f_c[0][0][0][s]) > 0.5:
                f_c[0][0][0][s] = 1
            if sigmoid(f_c[0][0][0][s]) <= 0.5:
                f_c[0][0][0][s] = 0

        guess = [ mx[0][part_id - 1],my[0][part_id - 1], f_c[0][0][0][part_id - 1] ]

        if f_c[0][0][0][part_id - 1] == visibility:
            visibility_correct += 1

        result = checker(image_id, part_id, guess, visibility)

        if result == True:
            correct += 1; total += 1

        if result == False:
            correct += 0; total += 1

    print "visibility score: ", visibility_correct / float(total)

    return correct / float( total )

x = tf.placeholder(tf.float32, shape = [1,224,224,3])
net = BIRDS_VGG_ILSVRC_16_layers({'data' : x}, trainable = True)

W = tf.Variable(tf.random_uniform([9,9,512,15],-1e-2,1e-2))
b = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))

conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv4_2'], W, [1,1,1,1], 'VALID'), b )
conv = tf.nn.relu(conv)

total = tf.reduce_sum(conv, [1,2], True)
total = tf.clip_by_value(total,1e-9,1000000000)
conv /= total

W1 = tf.Variable(tf.random_uniform([20,20,15,15],-1e-2,1e-2))
b1 = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))

fc = tf.nn.bias_add( tf.nn.conv2d(conv, W1, [1,1,1,1], 'VALID'), b1 )

guess = tf.sigmoid(fc)

mean_x, mean_y = 0,0

for i in range(20):
    for j in range(20):
        mean_x += conv[:,i,j,:] * (i + 0.5)
        mean_y += conv[:,i,j,:] * (j + 0.5)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, root2 + 'Experiments/Models/VGG_bird_visibility_model_conv4_9_trained')

for i in range(1,16):
    print "Part", str(i),":", model_accuracy(sess, i)
