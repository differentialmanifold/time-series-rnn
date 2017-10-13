from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.contrib import rnn
from reader import data_producer

# Training Parameters
learning_rate = 0.0001
display_step = 200

epoch_size = 2000
batch_size = 1
num_steps = 50
predict_len = 100
save_path = os.path.join(os.path.dirname(__file__), '../data/quant')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Network Parameters
num_input = 1
num_hidden = 128  # hidden layer num of features
num_output = 7

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_steps, num_input])
Y = tf.placeholder(tf.float32, [None, num_steps, num_input])
X_predict = tf.placeholder(tf.float32, [1, num_input])

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
    state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = lstm_cell(x[time_step], state)
            outputs.append(cell_output)

    # Linear activation, using rnn inner loop last output
    return tf.convert_to_tensor([tf.matmul(outputs[i], weights['out']) + biases['out'] for i in range(len(outputs))])


# def predict_new_point():
#     lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
#
#     state = lstm_cell.zero_state(1, tf.float32)
#     cell_input = X_predict
#
#     outputs = []
#     with tf.variable_scope("RNN"):
#         for time_step in range(predict_len):
#             tf.get_variable_scope().reuse_variables()
#             (cell_output, state) = lstm_cell(cell_input, state)
#
#             cell_input = tf.matmul(cell_output, weights['out']) + biases['out']
#             outputs.append(cell_input)
#     return outputs


# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, timesteps, n_input)
# Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
x_series = tf.unstack(X, num_steps, axis=1)
logits_series = RNN(x_series, weights, biases)
y_series = tf.unstack(Y, num_steps, axis=1)

# Define loss and optimizer
account = tf.convert_to_tensor([1.0, 0.0])
choose_options = tf.constant([(-1.0, 1), (-0.5, 1), (-0.25, 1), (0, 0), (0.25, 0), (0.5, 0), (1.0, 0)])
index = tf.argmax(logits_series, 1)
choosed = tf.gather(choose_options, index)
for i in range(len(y_series)):
    change = tf.gather(account, tf.to_int32(choosed[0][1])) * choosed[0][0]
    new_account1 = account[0] - change
    new_account2 = (account[1] + change) * (1 + y_series[i][0])
    account = tf.convert_to_tensor([new_account1, new_account2])
loss_op = -tf.log(tf.reduce_sum(account))

# loss_op = tf.reduce_sum(
#     tf.convert_to_tensor([tf.reduce_sum(tf.square(y_series[i] - logits_series[i])) for i in range(len(y_series))]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# predict_logits_series = predict_new_point()

sin_producer = data_producer.DataProducer(batch_size, num_steps)
x, y = sin_producer.sin_producer(epoch_size)
x = tf.reshape(x, [batch_size, num_steps, num_input])
y = tf.reshape(y, [batch_size, num_steps, num_output])

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
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
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
    finally:
        coord.request_stop()
        coord.join()

    saver.save(sess, os.path.join(save_path, 'model.ckpt'))
    print("Optimization Finished!")

    print('test logits_series is:')
    logits_value = sess.run(logits_series, feed_dict={X: test_x})
    loss = sess.run(loss_op, feed_dict={X: test_x, Y: test_y})
    print("Minibatch Loss= {:.4f}".format(loss))

    # predict_logits_value = sess.run(predict_logits_series, feed_dict={X_predict: [[0.0]]})

    x_points = np.linspace(0, num_steps * 0.1, num_steps)
    y_points = [item[-1][0] for item in logits_value]
    test_x_points = [item[0] for item in test_x[-1]]

    plt.plot(x_points, y_points, 'ro')
    plt.plot(x_points, test_x_points, 'bo')

    plt.show()

    # x_points = np.linspace(0, predict_len * 0.1, predict_len)
    # y_points = [item[0][0] for item in predict_logits_value]
    # plt.plot(x_points, y_points, 'ro')
    # plt.show()
