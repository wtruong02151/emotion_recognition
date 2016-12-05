import model
import sys
import tensorflow as tf
# Takes in a file path to a directory with *.jpg files, saves them into
# a format that TF can read and returns an array of image data + their labels
def read_in_data(file_path):
    ##################################### READ IN DATA #####################################
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    file_names = tf.train.match_filenames_once(file_path + "/*.jpg")
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

        # print len(file_names.eval())
        # print len(set(file_names.eval()))

        for i in range(len(file_names.eval())):
            # I think what happens when you call eval on either key or value, it will iterate both key and
            # val to the next item in the queue. I tried calling key.eval and value.eval both and that fucked it up.
            # Key here is the file's name. Value is actually the file.
            file_name = key.eval()
            labels = file_name.decode().split("-")

            if (i % 100 == 0 and i != 0):
                print i

            if file_name in seen:
                print("seen:") + file_name
            else:
                seen.add(file_name)

            #  [male, female]
            # labels[1][1] is race and sex
            # if labels[1][1] == 'M': #'AM' stands for Asian Male, but I only have AM and AF currently in the file
            #     lab.append([1, 0])
            # else: # 0 for girls
            #     lab.append([0, 1])

            # [Happy Open, Happy Closed, Neutral, Angry, Fear]
            image_emotion = labels[4][:-4].split('.')[0]
            if (image_emotion == 'HO'):
                lab.append([1, 0, 0, 0, 0])
            elif (image_emotion == 'HC'):
                lab.append([0, 1, 0, 0, 0])
            elif (image_emotion == 'N'):
                lab.append([0, 0, 1, 0, 0])
            elif (image_emotion == 'A'):
                lab.append([0, 0, 0, 1, 0])
            elif (image_emotion == 'F'):
                lab.append([0, 0, 0, 0, 1])
            else:
                print('ERROR: UNEXPECTED LABEL RECEIVED')
                sys.exit()

            # Read the in the image and decode it. It might be possible here to use "value" instead of,
            # but I'm not sure. You guys can mess with it.
            my_img =  tf.image.decode_jpeg(tf.read_file(file_name), channels=1)
            image = tf.image.resize_images(my_img, 31, 32)
            leh = tf.reshape(image, [1, 992])
            data.append(leh.eval()[0])

        coord.request_stop()
        coord.join(threads)

    print("Done reading images.")
    return data, lab

def train_and_test(data, lab, train_flag):
    length = len(data)
    split_index = int(length*.9)

    train_data = data[:split_index]
    train_labels = lab[:split_index]

    # Changethis to be equal to data/lab if you aren't training
    test_data = data[split_index:]
    test_labels = lab[split_index:]

    BATCHSZ = 10
    EPOCHS = 20
    BATCHES = len(train_data)/BATCHSZ

    test_data = data
    test_labels = lab

    with tf.variable_scope("convolutional"):
        x = tf.placeholder("float", [None, 992])
        keep_prob = tf.placeholder(tf.float32)
        y_conv, variables = model.convolutional(x, keep_prob)

    # train
    y_ = tf.placeholder("float", [None, 5])
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver(variables)
    init = tf.initialize_all_variables()

    # Path to checkpoint files
    checkpoint_path = "../static/"

    with tf.Session() as sess:
        sess.run(init)

        print("###########TRAINING###########")
        #You'll notice sometimes it learns a lot better than other times. Probably due to random initialization of weights
        for i in range(EPOCHS):

            for j in range(BATCHES):
                train_accuracy = accuracy.eval(feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob : 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                train_step.run(feed_dict={x: train_data[j*BATCHSZ:(j+1)*BATCHSZ], y_: train_labels[j*BATCHSZ:(j+1)*BATCHSZ], keep_prob : .5})

        # Save TF variables
        saver.save(sess, checkpoint_path + "emotion.ckpt")

        # for i in range(20000):
        #     batch = data.train.next_batch(50)
        #     if i % 100 == 0:
        #         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        #         print("step %d, training accuracy %g" % (i, train_accuracy))
        #     sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        #
        # print(sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
        #
        # path = saver.save(sess, os.path.join(os.path.dirname(__file__), "data/convolutional.ckpt"))
        # print("Saved:", path)


if __name__ == "__main__":
    if (len(sys.argv)) != 3:
        print("Requires two parameters: 1) file path to images, 2) 'T' or 'F' to indicate if the network should train or load from a checkpoint.")
        sys.exit(1)

    file_path = sys.argv[1]
    train_flag = True if sys.argv[2] == 'T' else False

    data, lab = read_in_data(file_path)
    train_and_test(data, lab, train_flag)
