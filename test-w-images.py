
# coding: utf-8

# In[2]:

# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np


# In[16]:

# Make a queue of file names including all the JPEG images files in the relative
# image directory.


# filename_queue = tf.train.string_input_producer(
#     tf.train.match_filenames_once('undercooked/*.jpeg'))

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
# image = tf.image.decode_png(image_file, channels=4)
image = tf.image.decode_jpeg(image_file, channels=1)
image = tf.image.resize_images(image, 300, 300)
leh = tf.reshape(image, [1, 90000])
# print(image)
# print(leh)
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
#     print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)


# In[11]:

# print(np.array(felege[0]).shape)
# print(np.array(felege[0]))
# print(len(felege[0][0]))
print(np.array([[0,1]]).shape)


# In[10]:

import tensorflow as tf
x = tf.placeholder(tf.float32, [1, 90000])
# convolution, flatten
W = tf.Variable(tf.zeros([90000, 2]))
b = tf.Variable(tf.zeros([2]))

# print(b.shape)


# In[17]:

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.int32, [1])
# print y
# print y_

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

err  = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(err)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# In[26]:


print len(felege[0][0])

# In[22]:

for i in range(3):
    print("batch num", i)
#   batch_xs, batch_ys = mnist.train.next_batch(100)
    t, bruce, prediction, c = sess.run([train_step, W, y, cross_entropy], feed_dict={x: [felege[0][0]], y_: [1]})
    print(bruce)
    print("Prediction", prediction)
    print("Cross Entropy",c)
    


# In[19]:

print(np.array([felege[0][0], felege[0][0], felege[0][0]]).shape)
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run([y], feed_dict={x: [felege[0][0]], y_: [1]}))


# In[ ]:



# 