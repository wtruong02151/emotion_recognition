
# coding: utf-8

# In[2]:

# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np


# In[16]:

# Make a queue of file names including all the JPEG images files in the relative
# image directory.


filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once('undercooked/*.jpg'))

fi = tf.train.match_filenames_once('undercooked/*.jpg')

# filenames = ['undercooked/green.jpg', 'undercooked/blue.jpg']
# filename_queue = tf.train.string_input_producer(filenames)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
key, value = image_reader.read(filename_queue)
# print value



# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
# my_image = tf.image.decode_jpeg(value, channels=1)
data = []
lab = []

seen = set()
# Start a new session to show example output.
with tf.Session() as sess:  
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    for i in range(20):
        file_name = key.eval()
        labels = file_name.split('-')
        my_img =  tf.image.decode_jpeg(tf.read_file(file_name), channels=1)
        image = tf.image.resize_images(my_img, 300, 300)
        leh = tf.reshape(image, [1, 90000])
        if labels[1] == 'AM':
            lab.append(1)
        else:
            lab.append(0)
        data.append(leh.eval()[0])
    coord.request_stop()
    coord.join(threads)
  


#################################################################
import tensorflow as tf
x = tf.placeholder(tf.float32, [5, 90000])

W = tf.Variable(tf.zeros([90000, 2]))
b = tf.Variable(tf.zeros([5, 2]))


y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.int32, [5])


pred = tf.argmax(y, 1)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

err  = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(err)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)



for i in range(200):
    # print("batch num", i)
    if i%4 == 3:
        continue
    t, bruce, prediction, c, error = sess.run([train_step, W, pred, cross_entropy, err], feed_dict={x: data[(i%4)*5:((i+1)%4)*5], y_: lab[(i%4)*5:((i+1)%4)*5]})

    



# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("predicted", sess.run([pred], feed_dict={x: data[15:], y_: lab[15:]}))
print("correct", lab[15:])






