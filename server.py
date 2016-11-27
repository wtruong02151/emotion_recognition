# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
from flask import Flask
app = Flask(__name__)


#Functions to create variables and bias' so we don't have to keep doing them
#Slight positive bias since we are using activation function ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#Functions to handle convolution and pooling as well. Stride size of 1 and padding so that
#the output is the same sizes our input.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#We are pooling over a 2x2 block.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

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
		labels = file_name.decode().split("-")

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
EPOCHS = 3
BATCHES = int(len(train_data)/BATCHSZ)

##################################### BUILD THE MODEL #####################################

x = tf.placeholder(tf.float32, [None, 90000])
y_ = tf.placeholder(tf.int32, [None])

#First Convolutional Layer 5x5 patches, 1 input channel, and 32 output channels
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1,300,300,1])

#Apply the first layer using convolution, activation function ReLU and maxpooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second convolutional layer. 5x5 patches, 32 inputs from h_pool1, output 64 features
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#Apply the second layer to the output of the first layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer
W_fc1 = weight_variable([75 * 75 * 64, 16]) # dimensions of size of h_pool2
b_fc1 = bias_variable([16])

h_pool2_flat = tf.reshape(h_pool2, [-1, 75 * 75 * 64]) # dimensions of size of h_pool2
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Reduce over fitting with dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Finish up with softmax in the readout layer
W_fc2 = weight_variable([16, 2])
b_fc2 = bias_variable([2])


# W = tf.Variable(tf.zeros([90000, 2]))
# b = tf.Variable(tf.zeros([2]))

# y = tf.matmul(x, W) + b

y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


pred = tf.argmax(y, 1)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)

err  = tf.reduce_mean(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(err)

##################################### TEST AND TRAIN #####################################

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)


	print("###########TRAINING###########")
	#You'll notice sometimes it learns a lot better than other times. Probably due to random initialization of weights
	for i in range(EPOCHS):
		for j in range(BATCHES):
			# bruce, prediction, c, error, output = sess.run([pred, cross_entropy, err, y], feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob: .5})
			t,prediction, output, error = sess.run([train_step,pred, y, err], feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob : 1})
			# print len(output)
			# print prediction

			if True:
				print("EPOCH:", i)
				print("BATCH:", j)
				print("ERROR:", error)
				# print "CROSS ENTROPY:", c
				# print "------------"


	print("###########TESTING###########")
	correct = 0
	predicted_labels = sess.run(pred, feed_dict={x: test_data, y_: test_labels, keep_prob : 1})
	correct_labels = test_labels
	print("PREDICTED LABELS:", list(predicted_labels))
	print("CORRECT LABELS:  ", correct_labels)
	print("------------")
	for k in range(len(predicted_labels)):
		if predicted_labels[k] == correct_labels[k]:
			correct += 1

	print("TEST ACCURACY:", (correct)/float(len(predicted_labels)))


@app.route("/")
def hello():
	return "Hello World!"

@app.route("/test")
def callNet():
	print('in call net')
	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)
		
		print("###########TESTING###########")
		wrong = 0
		test_increment = int(len(test_data)/BATCHSZ)
		for i in range(test_increment):
			predicted_labels = sess.run(pred, feed_dict={x: test_data[i*BATCHSZ:(i+1)*BATCHSZ], y_: test_labels[i*BATCHSZ:(i+1)*BATCHSZ]})
			correct_labels = test_labels[i*BATCHSZ:(i+1)*BATCHSZ]
			print("PREDICTED LABELS:", predicted_labels)
			print("CORRECT LABELS:", correct_labels)
			print("------------")
			for k in range(len(predicted_labels)):
				if predicted_labels[k] != correct_labels[k]:
					wrong += 1

		print("TEST ACCURACY:", (len(predicted_labels)*test_increment - wrong)/float(len(predicted_labels)*test_increment))


if __name__ == "__main__":
	app.run()
