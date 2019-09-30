import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from cnn_utils import *

def predict():
	X,_,keep_prob = create_placeholder(64, 64, 3, 6)
	output = forward_propagation(X, keep_prob)
	output = tf.argmax(output,1)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess,tf.train.latest_checkpoint("model/"))
		num = 1
		while(num < 10):
			my_image = "datasets/sample/" + str(num) + ".jpg"
			image = np.array(ndimage.imread(my_image, flatten=False))#.astype(np.float32)
			my_predicted_image = (image.reshape((1,64,64,3))/255).astype(np.float32)

			my_predicted_image = sess.run(output, feed_dict={X:my_predicted_image,keep_prob:1.0})

			plt.imshow(image) 
			print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
			plt.show()
			num = num + 1

if __name__ == '__main__':
	predict()


