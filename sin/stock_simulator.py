from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.contrib import rnn
from reader import dataset_producer

# Training Parameters
learning_rate = 0.0001
display_step = 600

epoch_size = 200
batch_size = 5
num_steps = 100
predict_len = 100
save_path = os.path.join(os.path.dirname(__file__), '../data/stock_simulator')
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Network Parameters
num_hidden = 128  # hidden layer num of features

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_steps, 1])
Y = tf.placeholder(tf.float32, [None, num_steps, 1])
X_predict = tf.placeholder(tf.float32, [1, 1])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, 1]))
}
biases = {
    'out': tf.Variable(tf.random_normal([1]))
}


def RNN(x, weights, biases):
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = lstm_cell(x[time_step], state)
            outputs.append(cell_output)

    # Linear activation, using rnn inner loop last output
    return tf.convert_to_tensor([tf.matmul(outputs[i], weights['out']) + biases['out'] for i in range(len(outputs))])


def predict_new_point():
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    state = lstm_cell.zero_state(1, tf.float32)
    cell_input = X_predict

    outputs = []
    with tf.variable_scope("RNN"):
        for time_step in range(predict_len):
            tf.get_variable_scope().reuse_variables()
            (cell_output, state) = lstm_cell(cell_input, state)

            cell_input = tf.matmul(cell_output, weights['out']) + biases['out']
            outputs.append(cell_input)
    return outputs


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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

predict_logits_series = predict_new_point()

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

    train_dataset, test_dataset = dataset_producer.DatasetProducer(batch_size, num_steps).producer()

    for step in range(1, epoch_size):
        iterator = train_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        values = sess.run(next_element)
        batch_x = values['features']
        batch_y = values['labels']

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss))

    saver.save(sess, os.path.join(save_path, 'model.ckpt'))
    print("Optimization Finished!")

    print('test logits_series is:')
    test_iterator = test_dataset.make_one_shot_iterator()
    test_next_element = test_iterator.get_next()
    test_values = sess.run(test_next_element)
    test_x = test_values['features']
    test_y = test_values['labels']
    logits_value = sess.run(logits_series, feed_dict={X: test_x})
    loss = sess.run(loss_op, feed_dict={X: test_x, Y: test_y})
    print("Minibatch Loss= {:.4f}".format(loss))

    predict_logits_value = sess.run(predict_logits_series, feed_dict={X_predict: [[0.0]]})

x_points = np.linspace(0, num_steps * 0.1, num_steps)
y_points = [item[-1][0] for item in logits_value]
test_x_points = [item[0] for item in test_x[-1]]

plt.plot(x_points, y_points, 'ro')
plt.plot(x_points, test_x_points, 'bo')

plt.show()

x_points = np.linspace(0, predict_len * 0.1, predict_len)
y_points = [item[0][0] for item in predict_logits_value]
plt.plot(x_points, y_points, 'ro')
plt.show()
