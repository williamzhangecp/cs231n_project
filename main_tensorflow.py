import numpy as np
import math
import matplotlib.pyplot as plt
import process_data
import random
import tensorflow as tf

import sys
import yaml

# Analyzing command line arguments
if len(sys.argv) < 2:
  print 'Usage:'
  print '  python %s <.csv file>' % sys.argv[0]
  exit()

csv_file = sys.argv[1]

# Later we will have a list of filters and so on, easier to define this in here
# parameters
num_train = 1000
num_val = 100
num_total = num_val + num_train
num_filters = 32
num_filters2 = 64
filter_size = 5
num_classes = 7
weight_scale = 0.002088691359
hidden_dim = 500
reg = 0.000511417045343
num_epochs = 5
batch_size = 64
learning_rate = 0.000135139484703 * 100000000
lr_decay = 0.95

# read in data
X_train, y_train, X_test, y_test, X_val, y_val = process_data.read_faces_csv(csv_file, num_total, use_tensorflow = True)

_, H, W, _ = X_train.shape

# Split data into training and validation
X_val = X_train[-num_val:,:,:,:]
y_val = y_train[-num_val:]

X_train = X_train[:num_train,:,:,:]
y_train = y_train[:num_train]

# Convert y to one hot
y_train_one_hot = np.zeros((num_train, num_classes))
y_train_one_hot[np.arange(num_train), y_train.astype(int)] = 1

y_val_one_hot = np.zeros((num_val, num_classes))
y_val_one_hot[np.arange(num_val), y_val.astype(int)] = 1

#data = {'X_train' : X_train[:num_train,:,:,:], 'y_train' : y_train[:num_train], \
#		"X_val" : X_val[:num_val,:,:,:], "y_val" : y_val[:num_val] }


## TODO
# Define a class here to build the computational graph, not a nice way to do it like this
##

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) * weight_scale
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape) * weight_scale
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



X = tf.placeholder(tf.float32, shape=[None, H, W, 1])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

# First conv/relu/pool layer (Size is 48x48)

W_conv1 = weight_variable( (filter_size, filter_size, 1, num_filters) )
b_conv1 = bias_variable( [num_filters] )
h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# Second conv/relu/pool layer (Size is 24x24 from max pool)

W_conv2 = weight_variable( (filter_size, filter_size, num_filters, num_filters2) )
b_conv2 = bias_variable( [num_filters2] )
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Affine/pool layer (Size is now 12x12)

hidden_dim_in = H/4 * H/4 * num_filters2

W_fc1 = weight_variable([hidden_dim_in, hidden_dim])
b_fc1 = bias_variable([hidden_dim])
h_pool2_flat = tf.reshape(h_pool2, [-1, hidden_dim_in])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Affine + softmax

W_fc2 = weight_variable([hidden_dim, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)


## TODO
# Define a "solver" class or something to wrap around all this crap
##

# Define softmax_loss like we did in class
softmax_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0))) # Avoid log(0)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(softmax_loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

# Want to see all data at every epoch
num_iters = int( math.floor(float(num_train)/batch_size) )

for n in range(num_epochs):

    random_indices = np.random.permutation(num_train)
    lower_bound = 0
    print "Epoch: ", n, "/", num_epochs

    for i in range(num_iters):
      # pick out random part of data for batch, but make sure you iterate over all data
      batch_indices = random_indices[lower_bound:(lower_bound + batch_size)]
      lower_bound += batch_size
      X_batch = X_train[batch_indices,:,:,:]
      y_batch = y_train_one_hot[batch_indices, :]

      # This is where the magic happens
      _, loss, b_test = sess.run([train_step, softmax_loss, b_conv1],
                               feed_dict={X: X_batch, y: y_batch})

      if i%10 == 0:
        print "iteration", i, "/", num_iters, "\t", "loss", loss
        # print b_test # to check if parameters are updating

# This will be useful when training large models saver = tf.train.Saver()
