import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

##################################### READ IN DATA #####################################

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
file_names = tf.train.match_filenames_once('me/*.jpg')
filename_queue = tf.train.string_input_producer(file_names)

# Used to read the entire file
file_reader = tf.WholeFileReader()

# Read a whole file from the queue. Not entirely sure how this works but it returns
# a key and value, that we can use to iterate through the entire file.
key, value = file_reader.read(filename_queue)

#Array to hold our image data and also our labels
data = []
lab = []

with tf.Session() as sess:

    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    seen = set()

    print len(file_names.eval())
    # print len(set(file_names.eval()))

    for i in range(len(file_names.eval())):
        # I think what happens when you call eval on either key or value, it will iterate both key and
        # val to the next item in the queue. I tried calling key.eval and value.eval both and that fucked it up.
        # Key here is the file's name. Value is actually the file.
        file_name = key.eval()
        labels = file_name.decode().split("-")

        if (i % 100 == 0):
            print(i)

        if file_name in seen:
            print("seen:") + file_name
        else:
            seen.add(file_name)

        #  [male, female]
        if labels[1][1] == 'M': #'AM' stands for Asian Male, but I only have AM and AF currently in the file
            lab.append([1, 0])
        else: # 0 for girls
            lab.append([0, 1])

        # Read the in the image and decode it. It might be possible here to use "value" instead of,
        # but I'm not sure. You guys can mess with it.
        my_img =  tf.image.decode_jpeg(tf.read_file(file_name), channels=1)
        image = tf.image.resize_images(my_img, 28, 28)
        leh = tf.reshape(image, [1, 784])
        data.append(leh.eval()[0])

    coord.request_stop()
    coord.join(threads)

print("done")
length = len(data)
split_index = int(length*.8)

train_data = data[:split_index]
train_labels = lab[:split_index]

# Changethis to be equal to data/lab if you aren't training
test_data = data[split_index:]
test_labels = lab[split_index:]

test_data = data
test_labels = lab

print (test_labels)

BATCHSZ = 10
EPOCHS = 20
BATCHES = len(train_data)/BATCHSZ


x = tf.placeholder(tf.float32, shape=[None, 784])
# y_ = tf.placeholder(tf.float32, shape=[None, 10])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
# W = tf.Variable(tf.zeros([784,10]))
W = tf.Variable(tf.zeros([784,2]))
# b = tf.Variable(tf.zeros([10]))
b = tf.Variable(tf.zeros([2]))
y = tf.matmul(x,W) + b

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Use TF's Saver class to save TF Variables and tensors (like our weights and biases)
saver = tf.train.Saver()
# Temp variable to determine if network should train or not (should be passed in as a parameter when running the script))
train = False
# Path to checkpoint files
checkpoint_path = "../emotion_recognition/"
##################################### TEST AND TRAIN #####################################

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    if (train):
        print("###########TRAINING###########")
        #You'll notice sometimes it learns a lot better than other times. Probably due to random initialization of weights
        for i in range(EPOCHS):

            # Save TF variables every two epochs
            if (i % 2 == 0):
                saver.save(sess, checkpoint_path + "my_model.ckpt", global_step=i)

            for j in range(BATCHES):
                train_accuracy = accuracy.eval(feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob : 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob : .5})
    else:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        # I think by default this loads the latest checkpoint... not sure though
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("No checkpoint found.")

    print "###########TESTING###########"
    print("test accuracy %g"%accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob : 1.0}))
