import tensorflow as tf
import utils

def localization_net_pool5(net):
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,441])
    z_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,2])

    W = tf.Variable(tf.random_uniform([6,6,256,441],0,1e-6))
    b = tf.Variable(tf.random_uniform([441],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['pool5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    res = tf.reshape(res, [utils.BATCH_SIZE,441])
    y = tf.sigmoid(res)
    y = tf.clip_by_value( y, 1e-10, 1.0-1e-10 )

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)
    z = tf.clip_by_value( z, 1e-10, 1.0-1e-10 )

    loss_classification = - tf.reduce_mean( z_ * tf.log(z) )
    loss_detection = - tf.reduce_mean( y_ * tf.log(y) + (1.-y_) * tf.log(1.-y) )
    loss = loss_detection + loss_classification

    correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return (y_,z_,y,z,loss,accuracy)

def localization_net_conv5(net):
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,8281])
    z_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,2])

    W = tf.Variable(tf.random_uniform([13,13,256,8281],0,1e-6))
    b = tf.Variable(tf.random_uniform([8281],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['conv5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    res = tf.reshape(res, [utils.BATCH_SIZE,8281])
    y = tf.sigmoid(res)

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    loss_classification = - tf.reduce_mean( z_ * tf.log(z) )
    loss_detection = - tf.reduce_mean( y_ * tf.log(y) + (1.-y_) * tf.log(1.-y) )
    loss = loss_detection + loss_classification

    correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return (y_,z_,y,z,loss,accuracy)

def localization_net_conv5_convolution(net):
    y_ = [[0 for j in range(14)] for i in range(14)]
    W = [[0 for j in range(14)] for i in range(14)]
    b = [[0 for j in range(14)] for i in range(14)]
    conv = [[0 for j in range(14)] for i in range(14)]
    y = [[0 for j in range(14)] for i in range(14)]
    loss_detection = [[0 for j in range(14)] for i in range(14)]

    z_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,2])

    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    total_loss_detection = 0
    total = 0

    for i in range(1,14):
        for j in range(1,14):
            y_[i][j] =  tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,14-i,14-j])
            W[i][j] = tf.Variable(tf.random_uniform([i,j,256,1],0,1e-6))
            b[i][j] = tf.Variable(tf.random_uniform([1],0,1e-6))
            conv[i][j] = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W[i][j], [1,1,1,1], 'VALID'), b[i][j] )
            y[i][j] = tf.sigmoid(tf.reshape(conv[i][j],[utils.BATCH_SIZE,14-i,14-j]))
            y[i][j] = tf.clip_by_value( y[i][j], 1e-10, 1.0-1e-10 )
            loss_detection[i][j] = - tf.reduce_mean( y_[i][j] * tf.log(y[i][j]) + (1.-y_[i][j]) * tf.log(1.-y[i][j]) )
            total_loss_detection += loss_detection[i][j]
            total += 1
    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)
    z = tf.clip_by_value( z, 1e-10, 1.0-1e-10 )

    total_loss_detection /= total #take the average of all detection losses

    loss_classification = - tf.reduce_mean( z_ * tf.log(z) )
    loss = total_loss_detection + loss_classification

    correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return (y_,z_,y,z,loss,accuracy)

def face_localization_net_conv5_convolution(net):
    y_ = [[0 for j in range(14)] for i in range(14)]
    W = [[0 for j in range(14)] for i in range(14)]
    b = [[0 for j in range(14)] for i in range(14)]
    conv = [[0 for j in range(14)] for i in range(14)]
    y = [[0 for j in range(14)] for i in range(14)]
    loss_detection = [[0 for j in range(14)] for i in range(14)]

    total_loss_detection = 0
    total = 0

    for i in range(1,14):
        for j in range(1,14):
            y_[i][j] =  tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,14-i,14-j])
            W[i][j] = tf.Variable(tf.random_uniform([i,j,256,1],0,1e-6))
            b[i][j] = tf.Variable(tf.random_uniform([1],0,1e-6))
            conv[i][j] = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W[i][j], [1,1,1,1], 'VALID'), b[i][j] )
            y[i][j] = tf.sigmoid(tf.reshape(conv[i][j],[utils.BATCH_SIZE,14-i,14-j]))
            y[i][j] = tf.clip_by_value( y[i][j], 1e-10, 1.0-1e-10 )
            loss_detection[i][j] = - tf.reduce_mean( y_[i][j] * tf.log(y[i][j]) + (1.-y_[i][j]) * tf.log(1.-y[i][j]) )
            total_loss_detection += loss_detection[i][j]
            total += 1

    total_loss_detection /= total

    return (y_,y,total_loss_detection)

def localization_net_with_hidden_layer(net):
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,441])
    z_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,2])

    W = tf.Variable(tf.random_uniform([6,6,256,1000],0,1e-6))
    b = tf.Variable(tf.random_uniform([1000],0,1e-6))
    W_1 = tf.Variable(tf.random_uniform([1000,441],0,1e-6))
    b_1 = tf.Variable(tf.random_uniform([441],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['pool5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    hidden_layer = tf.reshape(res, [utils.BATCH_SIZE,1000])
    output_hidden_layer = tf.nn.relu(hidden_layer)
    output = tf.matmul(output_hidden_layer,W_1) + b_1
    y = tf.sigmoid(output)

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    loss_classification = - tf.reduce_mean( z_ * tf.log(z) )
    loss_detection = - tf.reduce_mean( y_ * tf.log(y) + (1.-y_) * tf.log(1.-y) )
    loss = loss_detection + loss_classification

    correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return (y_,z_,y,z,loss,accuracy)

def test_net_pool5(net):

    W = tf.Variable(tf.random_uniform([6,6,256,441],0,1e-6))
    b = tf.Variable(tf.random_uniform([441],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['pool5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    res = tf.reshape(res, [1,441])
    y = tf.sigmoid(res)

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    return (y,z)

def test_net_conv5(net):

    W = tf.Variable(tf.random_uniform([13,13,256,8281],0,1e-6))
    b = tf.Variable(tf.random_uniform([8281],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['conv5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    res = tf.reshape(res, [1,8281])
    y = tf.sigmoid(res)

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    return (y,z)


def test_net_with_hidden_layer(net):

    W = tf.Variable(tf.random_uniform([6,6,256,1000],0,1e-6))
    b = tf.Variable(tf.random_uniform([1000],0,1e-6))
    W_1 = tf.Variable(tf.random_uniform([1000,441],0,1e-6))
    b_1 = tf.Variable(tf.random_uniform([441],0,1e-6))
    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    fc = tf.nn.conv2d(net.layers['pool5'], W, [1,1,1,1], 'VALID')
    res = tf.nn.bias_add(fc,b)
    hidden_layer = tf.reshape(res, [1,1000])
    output_hidden_layer = tf.nn.relu(hidden_layer)
    output = tf.matmul(output_hidden_layer,W_1) + b_1
    y = tf.sigmoid(output)

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    return (y,z)

def test_net_conv5_convolution(net):
    W = [[0 for j in range(14)] for i in range(14)]
    b = [[0 for j in range(14)] for i in range(14)]
    conv = [[0 for j in range(14)] for i in range(14)]
    y = [[0 for j in range(14)] for i in range(14)]

    W_cls = tf.Variable(tf.random_uniform([4096,2],0,1e-6))
    b_cls = tf.Variable(tf.random_uniform([2],0,1e-6))

    for i in range(1,14):
        for j in range(1,14):
            W[i][j] = tf.Variable(tf.random_uniform([i,j,256,1],0,1e-6))
            b[i][j] = tf.Variable(tf.random_uniform([1],0,1e-6))
            conv[i][j] = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W[i][j], [1,1,1,1], 'VALID'), b[i][j] )
            y[i][j] = tf.sigmoid(tf.reshape(conv[i][j],[1,14-i,14-j]))

    fc_cls = tf.matmul(net.layers['fc7'], W_cls) + b_cls
    z = tf.nn.softmax(fc_cls)

    return (y,z)

def face_test_net_conv5_convolution(net):
    W = [[0 for j in range(14)] for i in range(14)]
    b = [[0 for j in range(14)] for i in range(14)]
    conv = [[0 for j in range(14)] for i in range(14)]
    y = [[0 for j in range(14)] for i in range(14)]

    for i in range(1,14):
        for j in range(1,14):
            W[i][j] = tf.Variable(tf.random_uniform([i,j,256,1],0,1e-6))
            b[i][j] = tf.Variable(tf.random_uniform([1],0,1e-6))
            conv[i][j] = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W[i][j], [1,1,1,1], 'VALID'), b[i][j] )
            y[i][j] = tf.sigmoid(tf.reshape(conv[i][j],[1,14-i,14-j]))

    return y


def face_visual_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,4])
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,4])
    W = tf.Variable(tf.random_uniform([5,5,256,4],-1e-6,1e-6))
    b = tf.Variable(tf.random_uniform([4],-1e-6,1e-6))
    conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W, [1,1,1,1], 'VALID'), b )
    conv = tf.nn.relu(conv)

    total = tf.reduce_sum(conv, [1,2], True)
    total = tf.clip_by_value(total,1e-9,1000000000)
    conv /= total

    mean_x, mean_y = 0,0

    for i in range(9):
        for j in range(9):
            mean_x += conv[:,i,j,:] * (i + 0.5)
            mean_y += conv[:,i,j,:] * (j + 0.5)


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(9):
        for j in range(9):
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
    loss2 /= utils.BATCH_SIZE
    loss2 /= 4

    loss = tf.reduce_sum(loss)
    loss /= utils.BATCH_SIZE
    loss /= 4

    return loss, mean_x, mean_y, x_, y_, loss2



def bird_visual_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,15])
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,15])
    z_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,15])
    W = tf.Variable(tf.random_uniform([5,5,256,15],-1e-6,1e-6))
    b = tf.Variable(tf.random_uniform([15],-1e-6,1e-6))
    conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W, [1,1,1,1], 'VALID'), b )
    conv = tf.nn.relu(conv)

    total = tf.reduce_sum(conv, [1,2], True)
    total = tf.clip_by_value(total,1e-9,1000000000)
    conv /= total

    mean_x, mean_y = 0,0

    for i in range(9):
        for j in range(9):
            mean_x += conv[:,i,j,:] * (i + 0.5)
            mean_y += conv[:,i,j,:] * (j + 0.5)


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(9):
        for j in range(9):
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

    eta = 0.01
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 = tf.reduce_sum(loss2)
    loss2 /= utils.BATCH_SIZE
    loss2 /= 15

    loss = tf.reduce_sum(loss)
    loss /= utils.BATCH_SIZE
    loss /= 15

    return loss, mean_x, mean_y, x_, y_, z_, loss2

def face_five_point_detection_net(net):
    x_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,5])
    y_ = tf.placeholder(tf.float32, shape = [utils.BATCH_SIZE,5])
    W = tf.Variable(tf.random_uniform([5,5,256,5],-1e-6,1e-6))
    b = tf.Variable(tf.random_uniform([5],-1e-6,1e-6))
    conv = tf.nn.bias_add( tf.nn.conv2d(net.layers['conv5'], W, [1,1,1,1], 'VALID'), b )
    conv = tf.nn.relu(conv)

    total = tf.reduce_sum(conv, [1,2], True)
    total = tf.clip_by_value(total,1e-9,1000000000)
    conv /= total

    mean_x, mean_y = 0,0

    for i in range(9):
        for j in range(9):
            mean_x += conv[:,i,j,:] * (i + 0.5)
            mean_y += conv[:,i,j,:] * (j + 0.5)


    sxx, sxy, syy = 0.1,0,0.1

    for i in range(9):
        for j in range(9):
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

    eta = 0.01
    loss += eta * (sxx + syy)

    loss2 = tf.sqrt( (mean_x - x_) * (mean_x - x_) +
                     (mean_y - y_) * (mean_y - y_) )

    loss2 = tf.reduce_sum(loss2)
    loss2 /= utils.BATCH_SIZE
    loss2 /= 4

    loss = tf.reduce_sum(loss)
    loss /= utils.BATCH_SIZE
    loss /= 4

    return loss, mean_x, mean_y, x_, y_, loss2
