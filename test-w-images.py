# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filenames = ['undercooked/green.jpg']
filename_queue = tf.train.string_input_producer(filenames)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
_, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file, channels=1)
image = tf.image.resize_images(image, 300, 300)
leh = tf.reshape(image, [1, 90000])
felege = []

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    felege = sess.run([leh])
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

# Set up TF environment
x = tf.placeholder(tf.float32, [None, 90000])
# convolution, flatten
W = tf.Variable(tf.zeros([90000, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training
for i in range(2):
#   batch_xs, batch_ys = mnist.train.next_batch(100)
    t, error = sess.run([train_step, y], feed_dict={x: felege[0], y_: [[0, 1]]})
    print(error)

# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: [felege[0][0], felege[0][0], felege[0][0]], y_: [[0,1], [0,1], [0,1]]}))
