from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.contrib import rnn
from reader import data_producer

# Training Parameters
learning_rate = 0.0001
display_step = 200

epoch_size = 2000
batch_size = 10
num_steps = 50

# Network Parameters
num_input = 1  # MNIST data input (img shape: 28*28)
num_hidden = 128  # hidden layer num of features
num_output = 1  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_steps, num_input])
Y = tf.placeholder(tf.float32, [None, num_steps, num_output])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}


def RNN(x, weights, biases):
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.convert_to_tensor([tf.matmul(outputs[i], weights['out']) + biases['out'] for i in range(len(outputs))])
    # return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, timesteps, n_input)
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
x_series = tf.unstack(X, num_steps, axis=1)
logits_series = RNN(x_series, weights, biases)
y_series = tf.unstack(Y, num_steps, axis=1)

# Define loss and optimizer
loss_op = tf.reduce_sum(
    tf.convert_to_tensor([tf.reduce_sum(tf.square(y_series[i] - logits_series[i])) for i in range(len(y_series))]))
# loss_op = tf.reduce_sum(tf.square(y_series[-1] - logits_series))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

sin_producer = data_producer.DataProducer(batch_size, num_steps)
x, y = sin_producer.sin_producer(epoch_size)
x = tf.reshape(x, [batch_size, num_steps, num_input])
y = tf.reshape(y, [batch_size, num_steps, num_output])

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess, coord=coord)

    test_x = None
    test_y = None
    try:
        for step in range(epoch_size):
            if step == 0:
                test_x, test_y = sess.run([x, y])
                continue

            batch_x, batch_y = sess.run([x, y])
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                print(batch_y)
                print('logits_series is:')
                print(sess.run(logits_series, feed_dict={X: batch_x}))
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
    finally:
        coord.request_stop()
        coord.join()

    print("Optimization Finished!")

    print(test_y)
    print('test logits_series is:')
    logits_value = sess.run(logits_series, feed_dict={X: test_x})
    print(logits_value)
    loss = sess.run(loss_op, feed_dict={X: test_x, Y: test_y})
    print("Minibatch Loss= {:.4f}".format(loss))

    x_points = np.linspace(0, num_steps * 0.1, num_steps)
    y_points = [item[-1][0] for item in logits_value]
    test_x_points = [item[0] for item in test_x[-1]]

    plt.plot(x_points, y_points, 'ro')
    plt.plot(x_points, test_x_points, 'bo')

    plt.show()
