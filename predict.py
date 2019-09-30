import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from cnn_utils import *

model_path    = "model/"
datasets_file = "datasets/sample/"

def predict():
	X,_,keep_prob = create_placeholder(64, 64, 3, 6)
	output = forward_propagation(X, keep_prob)
	output = tf.argmax(output,1)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,tf.train.latest_checkpoint(model_path))
		for filename in os.listdir(datasets_file):
			if filename.endswith(".jpg") or filename.endswith(".png"):
				print(filename)
				my_image = filename
				image = np.array(ndimage.imread(datasets_file + my_image, flatten=False))#.astype(np.float32)
				my_predicted_image = (image.reshape((1,64,64,3))/255).astype(np.float32)

				my_predicted_image = sess.run(output, feed_dict={X:my_predicted_image,keep_prob:1.0})

				plt.imshow(image) 
				print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
				plt.show()


if __name__ == '__main__':
	predict()


