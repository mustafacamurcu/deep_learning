import tensorflow as tf
import numpy as np
import sys
import scipy.misc
import utils
import VGG_utils
import glob
import matplotlib.image as mpimg
import glob
import math
root = '/data/vision/torralba/health-habits/other/enes/'
sys.path.append(root + 'VGG_Face/')
sys.path.append('/afs/csail.mit.edu/u/k/kocabey/Desktop/caffe-tensorflow-master/')
from VGG_Face import VGG_Face

def mean_error(part_id,image_id):
    global sess, jpg, txt
    part_x = 2 * part_id
    part_y = 2 * part_id + 1
    f = open(txt[image_id])
    data = f.readlines()
    for i in range(len(data)):
        data[i] = int(data[i])
    inter_ocular_distance = math.sqrt((data[0] - data[2]) ** 2 + (data[1] - data[3]) ** 2)

    batch_x = np.zeros((1,224,224,3))
    batch_x[0,:,:,:,] = VGG_utils.image_preprocess(jpg[image_id])
    mx,my = sess.run([mean_x,mean_y], feed_dict = {x: batch_x})

    img = mpimg.imread(jpg[image_id])

    mx = mx * img.shape[0] / 42.
    my = my * img.shape[1] / 42.

    for i in range(5):
        mx[0][i] = int(mx[0][i])

    for i in range(5):
        my[0][i] = int(my[0][i])

    error = math.sqrt( (mx[0][part_id] - data[part_x]) ** 2 + (my[0][part_id] - data[part_y]) ** 2 )
    mean_error = error / inter_ocular_distance
    return mean_error

jpg = sorted(glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/SquareTest/Images/*.jpg'))
txt = sorted(glob.glob('/data/vision/torralba/health-habits/other/enes/CelebData/SquareTest/Points/*.txt'))

x = tf.placeholder(tf.float32, shape = [1,224,224,3])
net = VGG_Face({'data' : x}, trainable = True)

W = tf.Variable(tf.random_uniform([15,15,256,5],-1,1))
b = tf.Variable(tf.random_uniform([5],-1,1))
conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv3_2'], W, [1,1,1,1], 'VALID'), b )
conv = tf.nn.relu(conv)

total = tf.reduce_sum(conv, [1,2], True)
total = tf.clip_by_value(total,1e-9,1000000000)
conv /= total

mean_x, mean_y = 0,0

for i in range(42):
    for j in range(42):
        mean_x += conv[:,i,j,:] * (i + 0.5)
        mean_y += conv[:,i,j,:] * (j + 0.5)

saver = tf.train.Saver()
a = 0
sess = tf.Session()
saver.restore(sess, root + 'Experiments/Models/VGG_face_scratch_model_conv3_15_trained')

total = [0,0,0,0,0]
success = [0,0,0,0,0]

for i in range(len(jpg)):
    u = mean_error(0,i)
    total[0] += u
    if u <= 0.1:
        success[0] += 1

    u = mean_error(1,i)
    total[1] += u
    if u <= 0.1:
        success[1] += 1

    u = mean_error(2,i)
    total[2] += u
    if u <= 0.1:
        success[2] += 1

    u =  mean_error(3,i)
    total[3] += u
    if u <= 0.1:
        success[3] += 1

    u = mean_error(4,i)
    total[4] += u
    if u <= 0.1:
        success[4] += 1

    utils.show_progress(i,len(jpg))

print "\n"
for i in range(5):
    print total[i]/ float(len(jpg))

print "Mean Error"
print sum(total) / float(5 * len(jpg))

print "\n"

for i in range(5):
    print 1 - success[i] / float(len(jpg))

print "Failure Rate"
print 1 - sum(success) / float(5 * len(jpg))
