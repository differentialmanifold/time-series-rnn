from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class DataProducer(object):
    def __init__(self, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

    def sin_data(self, epoch_size, batch_size, num_steps):
        num_point = (epoch_size * num_steps + 1) * batch_size
        raw_data = np.linspace(0, 0.1 * num_point, num_point)
        return np.sin(raw_data)

    def sin_producer(self, epoch_size, name=None):
        raw_data = self.sin_data(epoch_size, self.batch_size, self.num_steps)

        return self.producer(raw_data, name)

    def producer(self, raw_data, name=None):
        batch_size = self.batch_size
        num_steps = self.num_steps

        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.float32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size

        data = tf.reshape(raw_data[0: batch_size * batch_len],
                          [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps

        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        with tf.name_scope(name, "DataProducer"):
            i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
            x = tf.strided_slice(data, [0, i * num_steps],
                                 [batch_size, (i + 1) * num_steps])
            x.set_shape([batch_size, num_steps])
            y = tf.strided_slice(data, [0, i * num_steps + 1],
                                 [batch_size, (i + 1) * num_steps + 1])
            y.set_shape([batch_size, num_steps])
            return x, y
