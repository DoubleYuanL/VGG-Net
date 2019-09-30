import math
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) 
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) 

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) 
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) 

    classes = np.array(test_dataset["list_classes"][:]) 
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def init_dataset():
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = load_dataset()
	X_train = X_train_orig / 255.
	X_test = X_test_orig / 255.
	Y_train = convert_to_one_hot(Y_train_orig, 6).T
	Y_test = convert_to_one_hot(Y_test_orig, 6).T
	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_test.shape[0])) 
	print ("X_train shape: " + str(X_train.shape)) 
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_test shape: " + str(X_test.shape))
	print ("Y_test shape: " + str(Y_test.shape))
	return X_train, Y_train, X_test, Y_test, classes

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
    Y = tf.placeholder(tf.float32, [None, n_y], name = "Y")
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

def forward_propagation(X, keep_prob):
	Z1 = tf.contrib.layers.conv2d(inputs=X,  num_outputs=64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z1 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=64, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z1 = tf.contrib.layers.max_pool2d(inputs=Z1, kernel_size=[2,2], stride=[2,2], padding='VALID')
	print(Z1.shape)
	Z2 = tf.contrib.layers.conv2d(inputs=Z1, num_outputs=128, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z2 = tf.contrib.layers.conv2d(inputs=Z2, num_outputs=128, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z2 = tf.contrib.layers.max_pool2d(inputs=Z2, kernel_size=[2,2], stride=[2,2], padding='VALID')
	print(Z2.shape)
	Z3 = tf.contrib.layers.conv2d(inputs=Z2, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z3 = tf.contrib.layers.conv2d(inputs=Z3, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z3 = tf.contrib.layers.conv2d(inputs=Z3, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z3 = tf.contrib.layers.conv2d(inputs=Z3, num_outputs=256, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z3 = tf.contrib.layers.max_pool2d(inputs=Z3, kernel_size=[2,2], stride=[2,2], padding='VALID')
	print(Z3.shape)
	Z4 = tf.contrib.layers.conv2d(inputs=Z3, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z4 = tf.contrib.layers.conv2d(inputs=Z4, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z4 = tf.contrib.layers.conv2d(inputs=Z4, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z4 = tf.contrib.layers.conv2d(inputs=Z4, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z4 = tf.contrib.layers.max_pool2d(inputs=Z4, kernel_size=[2,2], stride=[2,2], padding='VALID')
	print(Z4.shape)
	Z5 = tf.contrib.layers.conv2d(inputs=Z4, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z5 = tf.contrib.layers.conv2d(inputs=Z5, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z5 = tf.contrib.layers.conv2d(inputs=Z5, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	# Z5 = tf.contrib.layers.conv2d(inputs=Z5, num_outputs=512, kernel_size=[3,3], stride=[1,1], padding="SAME", activation_fn=tf.nn.relu)
	Z5 = tf.contrib.layers.max_pool2d(inputs=Z5, kernel_size=[2,2], stride=[2,2], padding='VALID')
	print(Z5.shape)
	Fa1 = tf.contrib.layers.flatten(Z5)
	F1 = tf.contrib.layers.fully_connected(Fa1,256,activation_fn=tf.nn.relu)
	D1 = tf.nn.dropout(F1, keep_prob)
	F2 = tf.contrib.layers.fully_connected(D1,128,activation_fn=tf.nn.relu)
	D2 = tf.nn.dropout(F2, keep_prob)
	output = tf.contrib.layers.fully_connected(D2,6,activation_fn=None)

	return output

def compute_loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
    return loss 