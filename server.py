# Typical setup to include TensorFlow.
import numpy as np
import tensorflow as tf
import sys
from base64 import decodestring
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image

sys.path.append('static')

app = Flask(__name__, static_url_path ='', template_folder='static/')

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

def crop_image(filename):
	original_image = Image.open(filename)
	width, height = original_image.size
	left, top, right, bottom = int(width*0.345), int(height*0.276), int(width*0.665), int(height*0.742)
	cropped_image = original_image.crop((left, top, right, bottom))
	cropped_image.save(filename)

def read_image(file_path):
	# Make a queue of file names including all the JPEG images files in the relative
	# image directory.
	file_names = tf.train.match_filenames_once(file_path)
	# Temp variable to determine if network should train or not (should be passed in as a parameter when running the script))
	filename_queue = tf.train.string_input_producer(file_names)
	# Used to read the entire file
	file_reader = tf.WholeFileReader()
	# Read a whole file from the queue. Not entirely sure how this works but it returns
	# a key and value, that we can use to iterate through the entire file.
	key, value = file_reader.read(filename_queue)
	#Array to hold our image data and also our labels
	data = []
	with tf.Session() as sess:
		tf.initialize_all_variables().run()

		# Coordinate the loading of image files.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for i in range(len(file_names.eval())):
			# I think what happens when you call eval on either key or value, it will iterate both key and
			# val to the next item in the queue. I tried calling key.eval and value.eval both and that fucked it up.
			# Key here is the file's name. Value is actually the file.
			file_name = key.eval()

			# Read the in the image and decode it. It might be possible here to use "value" instead of,
			# but I'm not sure. You guys can mess with it.
			my_img =  tf.image.decode_jpeg(tf.read_file(file_name), channels=1)
			image = tf.image.resize_images(my_img, 28, 28)
			leh = tf.reshape(image, [1, 784])
			data.append(leh.eval()[0])

		coord.request_stop()
		coord.join(threads)
	return data

@app.route("/")
def hello():
	return render_template('test.html')

@app.route('/api/conv', methods=['POST'])
def convnet():
	# Save the image
	image_path = 'submitted_images/selfie.jpg'
	image_data = request.json['data_uri'][23:] # 23 is the index in which the base 64 uri data starts
	image_to_save = open(image_path, 'wb')
	image_to_save.write(image_data.decode('base64'))
	image_to_save.close()

	# Crop the image and save it
	crop_image(image_path)

	data = read_image(image_path)

	###

	x = tf.placeholder(tf.float32, shape=[None, 784])
	# y_ = tf.placeholder(tf.float32, shape=[None, 10])
	y_ = tf.placeholder(tf.float32, shape=[None, 5])
	# W = tf.Variable(tf.zeros([784,10]))
	W = tf.Variable(tf.zeros([784,5]))
	# b = tf.Variable(tf.zeros([10]))
	b = tf.Variable(tf.zeros([5]))
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

	W_fc2 = weight_variable([1024, 5])
	b_fc2 = bias_variable([5])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	decision = tf.argmax(y_conv, 1)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	###

	saver = tf.train.Saver()
	checkpoint_path = "../emotion_recognition/static"
	##################################### TEST AND TRAIN #####################################
	with tf.Session() as sess:
		init = tf.initialize_all_variables()
		sess.run(init)

		checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
		# I think by default this loads the latest checkpoint... not sure though
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("###########TESTING###########")
			print(decision.eval(feed_dict={x: data, keep_prob : 1.0}))
			# sess.run(tf.argmax(y_conv,1), feed_dict={x: data, keep_prob : 1.0})
			return jsonify(results=[1])
		else:
			print('ERROR: chkpt not found')
			sys.exit(1)

if __name__ == "__main__":
	app.run()
