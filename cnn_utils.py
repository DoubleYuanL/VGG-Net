import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    print(train_set_x_orig.shape)
    print(train_set_y_orig.shape)

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

#创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
    Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

#初始化参数
def init_parameters():
    tf.set_random_seed(1) #指定随机种子
    W1 = tf.get_variable("W1",[3,3,3,6], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b1 = tf.get_variable("b1",[6,], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2",[3,3,6,12], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b2 = tf.get_variable("b2",[12,], initializer=tf.zeros_initializer())

    W3 = tf.get_variable("W3",[3,3,12,24], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b3 = tf.get_variable("b3",[24], initializer=tf.zeros_initializer())
    W4 = tf.get_variable("W4",[3,3,24,24], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4",[24], initializer=tf.zeros_initializer())

    W5 = tf.get_variable("W5",[3,3,24,48], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable("b5",[48], initializer=tf.zeros_initializer())
    W6 = tf.get_variable("W6",[3,3,48,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b6 = tf.get_variable("b6",[128], initializer=tf.zeros_initializer())
    W7 = tf.get_variable("W7",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b7 = tf.get_variable("b7",[128], initializer=tf.zeros_initializer())

    W8 = tf.get_variable("W8",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b8 = tf.get_variable("b8",[128], initializer=tf.zeros_initializer())
    W9 = tf.get_variable("W9",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b9 = tf.get_variable("b9",[128], initializer=tf.zeros_initializer())
    W10 = tf.get_variable("W10",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b10 = tf.get_variable("b10",[128], initializer=tf.zeros_initializer())

    W11 = tf.get_variable("W11",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b11 = tf.get_variable("b11",[128], initializer=tf.zeros_initializer())
    W12 = tf.get_variable("W12",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b12 = tf.get_variable("b12",[128], initializer=tf.zeros_initializer())
    W13 = tf.get_variable("W13",[3,3,128,128], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b13 = tf.get_variable("b13",[128], initializer=tf.zeros_initializer())

    parameters = {
                    "W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5,
                    "W6": W6, "b6": b6, "W7": W7, "b7": b7, "W8": W8, "b8": b8, "W9": W9, "b9": b9, "W10": W10, "b10": b10,
                    "W11": W11, "b11": b11, "W12": W12, "b12": b12, "W13": W13, "b13": b13} 
    return parameters

#前向传播
def forward_propagation(X, parameters,keep_prob):
    W1 = parameters['W1'] 
    b1 = parameters['b1'] 
    W2 = parameters['W2'] 
    b2 = parameters['b2'] 
    W3 = parameters['W3'] 
    b3 = parameters['b3']
    W4 = parameters['W4'] 
    b4 = parameters['b4']
    W5 = parameters['W5'] 
    b5 = parameters['b5']
    W6 = parameters['W6'] 
    b6 = parameters['b6'] 
    W7 = parameters['W7'] 
    b7 = parameters['b7'] 
    W8 = parameters['W8'] 
    b8 = parameters['b8']
    W9 = parameters['W9'] 
    b9 = parameters['b9']
    W10 = parameters['W10'] 
    b10 = parameters['b10']
    W11 = parameters['W11'] 
    b11 = parameters['b11'] 
    W12 = parameters['W12'] 
    b12 = parameters['b12'] 
    W13 = parameters['W13'] 
    b13 = parameters['b13']

    Z1 = tf.nn.bias_add(tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME"),b1)
    A1 = tf.nn.relu(Z1)
    print(A1.shape)
    Z2 = tf.nn.bias_add(tf.nn.conv2d(A1,W2,strides=[1,1,1,1],padding="SAME"),b2)
    A2 = tf.nn.relu(Z2)
    # lrn2 = tf.nn.lrn(A2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn2")
    P1 = tf.nn.max_pool(A2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    Z3 = tf.nn.bias_add(tf.nn.conv2d(P1,W3,strides=[1,1,1,1],padding="SAME"),b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.nn.bias_add(tf.nn.conv2d(A3,W4,strides=[1,1,1,1],padding="SAME"),b4)
    A4 = tf.nn.relu(Z4)
    P2 = tf.nn.max_pool(A4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    Z5 = tf.nn.bias_add(tf.nn.conv2d(P2,W5,strides=[1,1,1,1],padding="SAME"),b5)
    A5 = tf.nn.relu(Z5)
    Z6 = tf.nn.bias_add(tf.nn.conv2d(A5,W6,strides=[1,1,1,1],padding="SAME"),b6)
    A6 = tf.nn.relu(Z6)
    Z7 = tf.nn.bias_add(tf.nn.conv2d(A6,W7,strides=[1,1,1,1],padding="SAME"),b7)
    A7 = tf.nn.relu(Z7)
    P3 = tf.nn.max_pool(A7,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    Z8 = tf.nn.bias_add(tf.nn.conv2d(P3,W8,strides=[1,1,1,1],padding="SAME"),b8)
    A8 = tf.nn.relu(Z8)
    Z9 = tf.nn.bias_add(tf.nn.conv2d(A8,W9,strides=[1,1,1,1],padding="SAME"),b9)
    A9 = tf.nn.relu(Z9)
    Z10 = tf.nn.bias_add(tf.nn.conv2d(A9,W10,strides=[1,1,1,1],padding="SAME"),b10)
    A10 = tf.nn.relu(Z10)
    P4 = tf.nn.max_pool(A10,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    Z11 = tf.nn.bias_add(tf.nn.conv2d(P4,W11,strides=[1,1,1,1],padding="SAME"),b11)
    A11 = tf.nn.relu(Z11)
    Z12 = tf.nn.bias_add(tf.nn.conv2d(A11,W12,strides=[1,1,1,1],padding="SAME"),b12)
    A12 = tf.nn.relu(Z12)
    Z13 = tf.nn.bias_add(tf.nn.conv2d(A12,W13,strides=[1,1,1,1],padding="SAME"),b13)
    A13 = tf.nn.relu(Z13)
    P5 = tf.nn.max_pool(A13,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    print(P5.shape)

    #f1
    Fa1 = tf.contrib.layers.flatten(P5)
    F1 = tf.contrib.layers.fully_connected(Fa1,128,activation_fn=tf.nn.relu)#None)#tf.nn.relu) tf.nn.sigmoid
    D1 = tf.nn.dropout(F1, keep_prob)
    F2 = tf.contrib.layers.fully_connected(D1,64,activation_fn=tf.nn.relu)
    D2 = tf.nn.dropout(F2, keep_prob)
    Z16 = tf.contrib.layers.fully_connected(D2,6,activation_fn=None)#tf.nn.softmax)

    return Z16

#计算loss
def compute_loss(Z6,Y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z6, labels = Y))
    return loss 