
# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np

##################################### READ IN DATA #####################################
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once('undercooked/*.jpg'))

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
    
    # Right now there are 20 images in our file "uncooked"
    for i in range(20):
        # I think what happens when you call eval on either key or value, it will iterate both key and 
        # val to the next item in the queue. I tried calling key.eval and value.eval both and that fucked it up.
        # Key here is the file's name. Value is actually the file.
        file_name = key.eval()
        labels = file_name.split('-')
        if labels[1] == 'AM': #'AM' stands for Asian Male, but I only have AM and AF currently in the file
            lab.append(1)
        else: # 0 for girls
            lab.append(0)

        # Read the in the image and decode it. It might be possible here to use "value" instead of,
        # but I'm not sure. You guys can mess with it.
        my_img =  tf.image.decode_jpeg(tf.read_file(file_name), channels=1)
        image = tf.image.resize_images(my_img, 300, 300)
        leh = tf.reshape(image, [1, 90000])
        data.append(leh.eval()[0])

    coord.request_stop()
    coord.join(threads)
  


##################################### TRAIN AND TEST #####################################

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


#Weird indexing to make it train repeatedly on the first 15 datapts. The 200 batches here is arbitrary

#You'll notice sometimes it learns a lot better than other times. Probably due to random initialization of weights
for i in range(200):
    #For indexing purposes
    if i%4 == 3:
        continue
    t, bruce, prediction, c, error = sess.run([train_step, W, pred, cross_entropy, err], feed_dict={x: data[(i%4)*5:((i+1)%4)*5], y_: lab[(i%4)*5:((i+1)%4)*5]})
    
    if i%30 == 0:
        print "batch", i
        print "error", error
        print "------------"
    
#Test on the last 5 datapts. 
print "predicted", sess.run(pred, feed_dict={x: data[15:], y_: lab[15:]})
print "correct", lab[15:]






