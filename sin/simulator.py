from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import rnn
from reader import data_producer

# Training Parameters
learning_rate = 0.001
display_step = 200

epoch_size = 1000
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
x_series = tf.unstack(X, num_steps, num_input)
logits_series = RNN(x_series, weights, biases)
y_series = tf.unstack(Y, num_steps, num_input)

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

    try:
        for step in range(epoch_size):
            batch_x, batch_y = sess.run([x, y])
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op, feed_dict={X: batch_x,
                                                    Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
    finally:
        coord.request_stop()
        coord.join()

    print("Optimization Finished!")

# if __name__ == '__main__':
#     epoch_size = 100
#     batch_size = 10
#     num_steps = 50
#
#     sin_producer = data_producer.DataProducer(batch_size, num_steps)
#     x, y = sin_producer.sin_producer(epoch_size)
#     with tf.Session() as session:
#         coord = tf.train.Coordinator()
#         tf.train.start_queue_runners(session, coord=coord)
#         try:
#             xval, yval = session.run([x, y])
#             print(xval, yval)
#             xval, yval = session.run([x, y])
#             print(xval, yval)
#         finally:
#             coord.request_stop()
#             coord.join()
