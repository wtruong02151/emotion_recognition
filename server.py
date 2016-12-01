# Typical setup to include TensorFlow.
import tensorflow as tf
import numpy as np
from flask import Flask
app = Flask(__name__)


@app.route("/")
def hello():
	return "Hello World!"


@app.route('/api/conv', methods=['POST'])
def convnet():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        # I think by default this loads the latest checkpoint... not sure though
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("ERROR: No checkpoint found.")
            sys.exit()

        print "###########TESTING###########"
        print("test accuracy %g"%accuracy.eval(feed_dict={x: [input], keep_prob: 1.0}))
        return jsonify(results=[decision])


if __name__ == "__main__":
	app.run()
