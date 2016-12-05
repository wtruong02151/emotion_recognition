import tensorflow as tf

# Multilayer Convolutional Network
def convolutional(x, keep_prob):
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # First Convolutional Layer
    x_image = tf.reshape(x, [-1,31,32,1])
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Third Convolutional Layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    # Densely Connected Layer
    W_fc1 = weight_variable([4 * 4 * 128, 2048])
    b_fc1 = bias_variable([2048])
    h_pool3_flat = tf.reshape(h_pool3, [-1, 4*4*128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer
    W_fc2 = weight_variable([2048, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
