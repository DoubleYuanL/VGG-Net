import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import cnn_utils
import cv2
from cnn_utils import *

def predict():
	X,_,keep_prob = create_placeholder(64, 64, 3, 6)

	parameters = init_parameters()

	Z5 = forward_propagation(X, parameters, keep_prob)

	Z5 = tf.argmax(Z5,1)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,tf.train.latest_checkpoint("model/"))

		#use the sample picture to predict the unm
		sample = 1
		cam = 1
		num = 1
		if (sample):
			while(num < 10):
				my_image = "sample/" + str(num) + ".jpg"	
				num_px = 64
				fname =  my_image 
				image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
				my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
				my_predicted_image = my_predicted_image.astype(np.float32)

				my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image,keep_prob:1.0})

				plt.imshow(image) 
				print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
				plt.show()
				num = num + 1

		elif(cam):# use the camera to predict the num
			cap = cv2.VideoCapture(0)
			num = 0
			while (1):
				ret, frame = cap.read()
				cv2.namedWindow("capture")
				cv2.imshow("capture", frame)
				k = cv2.waitKey(1) & 0xFF
				if  k == ord('s'):
					frame = cv2.resize(frame, (int(256), int(256)))
					cv2.imwrite("sample/cam/" + str(num)+".jpg", frame)

					my_image = "sample/cam/" + str(num) + ".jpg"	
					num_px = 64
					fname =  my_image 
					image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
					my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
					my_predicted_image = my_predicted_image.astype(np.float32)

					my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image,keep_prob:1.0})


					# plt.imshow(image) 
					print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
					plt.show()
					num = num + 1
				elif k == ord('q'):
					break
			cap.release()
			cv2.destroyAllWindows()

if __name__ == '__main__':
	predict()


