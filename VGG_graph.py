import tensorflow as tf
import VGG_utils

def VGG_bird_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    z_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    W = tf.Variable(tf.random_uniform([5,5,512,15],-1e-2,1e-2))
    b = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))

    W1 = tf.Variable(tf.random_uniform([14,14,512,15],-1e-2,1e-2))
    b1 = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))

    fc = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5_2'], W1, [1,1,1,1], 'VALID'), b1 )

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

    sxx, sxy, syy = 0.1,0,0.1

    for i in range(10):
        for j in range(10):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    loss *= z_ #handling non-existent points

    #variance should also be penalized, otherwise it does not learn anything useful.

    eta = 0.001
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 *= z_

    loss2 = tf.reduce_sum(loss2)
    loss2 /= 15 * VGG_utils.BATCH_SIZE - tf.reduce_sum(1 - z_)

    loss = tf.reduce_sum(loss)


    delta = 100.
    loss +=  delta * tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits(fc,z_) ) # for guessing visibility

    loss /= VGG_utils.BATCH_SIZE

    return loss, mean_x, mean_y, x_, y_, z_, loss2

def VGG_bird_point_detection_net_conv4_9(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    z_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    W = tf.Variable(tf.random_uniform([9,9,512,15],-1e-2,1e-2))
    b = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))

    conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv4_2'], W, [1,1,1,1], 'VALID'), b )
    conv = tf.nn.relu(conv)

    total = tf.reduce_sum(conv, [1,2], True)
    total = tf.clip_by_value(total,1e-9,1000000000)
    conv /= total

    mean_x, mean_y = 0,0

    for i in range(20):
        for j in range(20):
            mean_x += conv[:,i,j,:] * (i + 0.5)
            mean_y += conv[:,i,j,:] * (j + 0.5)

    sxx, sxy, syy = 0.1,0,0.1

    for i in range(20):
        for j in range(20):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    loss *= z_ #handling non-existent points

    #variance should also be penalized, otherwise it does not learn anything useful.

    eta = 0.001
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 *= z_

    loss2 = tf.reduce_sum(loss2)
    loss2 /= 15 * VGG_utils.BATCH_SIZE - tf.reduce_sum(1 - z_)

    loss = tf.reduce_sum(loss)

    loss /= VGG_utils.BATCH_SIZE

    return loss, mean_x, mean_y, x_, y_, z_, loss2

def VGG_face_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,5])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,5])
    W = tf.Variable(tf.random_uniform([5,5,512,5],-1e-2,1e-2))
    b = tf.Variable(tf.random_uniform([5],-1e-2,1e-2))
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


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(10):
        for j in range(10):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    #variance should also be penalized, otherwise it does not learn anything useful.

    eta = 0.001
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 = tf.reduce_sum(loss2)
    loss2 /= VGG_utils.BATCH_SIZE
    loss2 /= 5.
    loss = tf.reduce_sum(loss)
    loss /= VGG_utils.BATCH_SIZE
    loss /= 5.

    return loss, mean_x, mean_y, x_, y_, loss2

def VGG_face_caltech_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,4])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,4])
    W = tf.Variable(tf.random_uniform([5,5,512,4],-1e-2,1e-2))
    b = tf.Variable(tf.random_uniform([4],-1e-2,1e-2))
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


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(10):
        for j in range(10):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    #variance should also be penalized, otherwise it does not learn anything useful.

    eta = 0.001
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 = tf.reduce_sum(loss2)
    loss2 /= VGG_utils.BATCH_SIZE
    loss2 /= 4.
    loss = tf.reduce_sum(loss)
    loss /= VGG_utils.BATCH_SIZE
    loss /= 4.

    return loss, mean_x, mean_y, x_, y_, loss2

def VGG_bird_multilayer(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])
    z_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,15])

    W = {}; b = {}; conv = {}; total = {}; mean_x = {}; mean_y = {}; M = {};
    filter_size  = [0,0,0,17,9,5]
    heatmap_size = [0,0,0,40,20,10]
    conv_size = [0,0,0,256,512,512]

    pred_x, pred_y = 0,0

    for layer in range(3,6):
        W[layer] = tf.Variable(tf.random_uniform([filter_size[layer],filter_size[layer],conv_size[layer],15],-1e-2,1e-2))
        b[layer] = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))
        M[layer] = tf.Variable(tf.random_uniform([15],-1e-2,1e-2))
        conv[layer] = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv' + str(layer) + '_2'], W[layer], [1,1,1,1], 'VALID'), b[layer] )
        conv[layer] = tf.nn.relu(conv[layer])

        total[layer] = tf.reduce_sum(conv[layer], [1,2], True)
        total[layer] = tf.clip_by_value(total[layer],1e-9,1000000000)
        conv[layer] /= total[layer]

        mean_x[layer], mean_y[layer] = 0,0

        for i in range(heatmap_size[layer]):
            for j in range(heatmap_size[layer]):
                mean_x[layer] += conv[layer][:,i,j,:] * (i + 0.5)
                mean_y[layer] += conv[layer][:,i,j,:] * (j + 0.5)

        mean_x[layer] /= heatmap_size[layer]
        mean_y[layer] /= heatmap_size[layer]

        u = []

        for i in range(VGG_utils.BATCH_SIZE):
            u.append(M[layer])

        S = tf.concat(0,u)
        S = tf.reshape(S,[VGG_utils.BATCH_SIZE,15])

        pred_mean_x = mean_x[layer] * tf.nn.softmax( S )
        pred_x += pred_mean_x

        pred_mean_y = mean_y[layer] * tf.nn.softmax( S )
        pred_y += pred_mean_y

    pred_x /= 3.
    pred_y /= 3.

    loss = (pred_x - x_) * (pred_x - x_) + \
           (pred_y - y_) * (pred_y - y_)

    # loss *= z_ #handling non-existent points

    loss2 = tf.sqrt( (pred_x - x_) * (pred_x - x_) +
                     (pred_y - y_) * (pred_y - y_) )

    # loss2 *= z_

    loss2 = tf.reduce_sum(loss2)
    loss2 /= VGG_utils.BATCH_SIZE
    # loss2 /= 15 - tf.reduce_sum(z_)

    loss = tf.reduce_sum(loss)
    loss /= VGG_utils.BATCH_SIZE
    # loss /= 15 - tf.reduce_sum(z_)

    return loss, pred_x, pred_y, x_, y_, z_, loss2

def VGG_face_scratch_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,5])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,5])
    W = tf.Variable(tf.random_uniform([15,15,256,5],-1e-2,1e-2))
    b = tf.Variable(tf.random_uniform([5],-1e-2,1e-2))
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


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(42):
        for j in range(42):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    #variance should also be penalized, otherwise it does not learn anything useful.

    eta = 0.001
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 = tf.reduce_sum(loss2)
    loss2 /= VGG_utils.BATCH_SIZE
    loss2 /= 5.
    loss = tf.reduce_sum(loss)
    loss /= VGG_utils.BATCH_SIZE
    loss /= 5.

    return loss, mean_x, mean_y, x_, y_, loss2

def VGG_human_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,14])
    y_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,14])
    z_ = tf.placeholder(tf.float32, shape = [VGG_utils.BATCH_SIZE,14])
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

    sxx, sxy, syy = 0.1,0,0.1

    for i in range(10):
        for j in range(10):
            sxx += conv[:,i,j,:] * (i - mean_x) * (i - mean_x)
            sxy += conv[:,i,j,:] * (i - mean_x) * (j - mean_y)
            syy += conv[:,i,j,:] * (j - mean_y) * (j - mean_y)

    k = 1. / (sxx * syy - sxy * sxy)
    a =  syy * k
    b = -sxy * k
    d =  sxx * k

    loss = a * (mean_x - x_) * (mean_x - x_) + \
           d * (mean_y - y_) * (mean_y - y_) + \
           2 * b * (mean_x - x_) * (mean_y - y_)

    loss *= z_ #handling non-existent points

    #variance should also be penalized, otherwise it does not learn anything useful.

    #eta = 0.001
    #loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 *= z_

    loss2 = tf.reduce_sum(loss2)
    loss2 /= 14 * VGG_utils.BATCH_SIZE - tf.reduce_sum(1 - z_)

    loss = tf.reduce_sum(loss)

    loss /= VGG_utils.BATCH_SIZE

    return loss, mean_x, mean_y, x_, y_, z_, loss2
