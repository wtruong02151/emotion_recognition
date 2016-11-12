
# coding: utf-8

# In[2]:

# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np


# In[16]:

# Make a queue of file names including all the JPEG images files in the relative
# image directory.


filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("undercooked/*.jpg"))

# filenames = ['undercooked/green.jpg', 'undercooked/blue.jpg']
# filename_queue = tf.train.string_input_producer(filenames)

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
key, value = image_reader.read(filename_queue)


# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
my_image = tf.image.decode_jpeg(value, channels=1)
all_imgs = []

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(20):
        image = my_image.eval()
        # print len(image)
        image = tf.image.resize_images(image, 300, 300)
        leh = tf.reshape(image, [1, 90000])
        all_imgs.append(leh.eval())   
    

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
# print all_imgs
# print len(all_imgs)
# for i in all_imgs:
    # print i
# labels1 = [1]*10
# labels2 = [0]*10
# labels  =labels1+labels2
# print labels



# In[10]:
#################################################################
import tensorflow as tf
x = tf.placeholder(tf.float32, [5, 90000])
# convolution, flatten
W = tf.Variable(tf.zeros([90000, 2]))
b = tf.Variable(tf.zeros([5, 2]))

# print(b.shape)


# In[17]:

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.int32, [5])


pred = tf.argmax(y, 1)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

err  = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(err)
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


# In[26]:




# In[22]:
batch_1 = [all_imgs[0][0], all_imgs[11][0], all_imgs[2][0], all_imgs[13][0], all_imgs[14][0]]
batch_2 = [all_imgs[10][0], all_imgs[1][0], all_imgs[12][0], all_imgs[3][0], all_imgs[4][0]]
label_1 = [1,0,1,0,0]
label_2 = [0,1,0,1,1]
for i in range(150):
    print("batch num", i)
    if i%2:
        t, bruce, prediction, c, error = sess.run([train_step, W, pred, cross_entropy, err], feed_dict={x: batch_1, y_: label_1})
        print("Cross Entropy",error)
    else:
        t, bruce, prediction, c, error = sess.run([train_step, W, pred, cross_entropy, err], feed_dict={x: batch_2, y_: label_2})
        print("Cross Entropy",error)
    
    
    


# In[19]:
test_data = [all_imgs[6][0], all_imgs[15][0], all_imgs[8][0], all_imgs[16][0], all_imgs[7][0]]
test_labels = [1, 0, 1, 0, 1]

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run([pred], feed_dict={x: test_data, y_: test_labels}))






