
# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np

##################################### READ IN DATA #####################################

# Make a queue of file names including all the JPEG images files in the relative
# image directory.
file_names = tf.train.match_filenames_once('undercooked/*.jpg')
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
    
    # seen = set()

    # print len(file_names.eval())
    # print len(set(file_names.eval()))

    for i in range(len(file_names.eval())):
        # I think what happens when you call eval on either key or value, it will iterate both key and 
        # val to the next item in the queue. I tried calling key.eval and value.eval both and that fucked it up.
        # Key here is the file's name. Value is actually the file.
        file_name = key.eval()
        labels = file_name.split('-')

        # if file_name in seen:
            # print "seen:" +f ile_name
        # else:
            # seen.add(file_name)

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

length = len(data)
split_index = int(length*.8)

train_data = data[:split_index]
train_labels = lab[:split_index]

test_data = data[split_index:]
test_labels = lab[split_index:]

BATCHSZ = 10
EPOCHS = 40
BATCHES = len(train_data)/BATCHSZ    

##################################### BUILD THE MODEL #####################################

x = tf.placeholder(tf.float32, [BATCHSZ, 90000])

W = tf.Variable(tf.zeros([90000, 2]))
b = tf.Variable(tf.zeros([BATCHSZ, 2]))


y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.int32, [BATCHSZ])

pred = tf.argmax(y, 1)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

err  = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(err)

##################################### TEST AND TRAIN #####################################

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)


    print "###########TRAINING###########"
    #You'll notice sometimes it learns a lot better than other times. Probably due to random initialization of weights
    for i in range(EPOCHS):
        for j in range(BATCHES):

            t, bruce, prediction, c, error = sess.run([train_step, W, pred, cross_entropy, err], feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ]})
            
            if j%100 == 0:
                print "EPOCH:", i
                print "BATCH:", j
                print "ERROR:", error
                print "------------"
    

    print "###########TESTING###########"
    wrong = 0
    test_increment = len(test_data)/BATCHSZ
    for i in range(test_increment):
        predicted_labels = sess.run(pred, feed_dict={x: test_data[i*BATCHSZ:(i+1)*BATCHSZ], y_: test_labels[i*BATCHSZ:(i+1)*BATCHSZ]})
        correct_labels = test_labels[i*BATCHSZ:(i+1)*BATCHSZ]
        print "PREDICTED LABELS:", predicted_labels
        print "CORRECT LABELS:", correct_labels
        print "------------"
        for k in range(len(predicted_labels)):
            if predicted_labels[k] != correct_labels[k]:
                wrong += 1

    print "TEST ACCURACY:", (len(predicted_labels)*test_increment - wrong)/float(len(predicted_labels)*test_increment)








