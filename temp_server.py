# Typical setup to include TensorFlow.
import model
import numpy as np
import tensorflow as tf
import sys
from base64 import decodestring
from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image

sys.path.append('static')

app = Flask(__name__, static_url_path ='', template_folder='static/')

x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
    checkpoint = tf.train.get_checkpoint_state("../emotion_recognition/static/")
    saver = tf.train.Saver()

if (checkpoint):
    saver.restore(sess, checkpoint.model_checkpoint_path)
else:
    print("fock me")

def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()

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
def main():
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

    # Grab image data and place it into a format that TF can read it.
    data = read_image(image_path)

    ##################################### TEST AND TRAIN #####################################
    print(convolutional(data))

if __name__ == "__main__":
    app.run()
