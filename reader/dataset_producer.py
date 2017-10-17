import numpy as np
import tensorflow as tf
import pandas as pd


class DatasetProducer(object):
    def __init__(self, batch_size, num_steps):
        self.batch_size = batch_size
        self.num_steps = num_steps

    def producer(self):
        data_path = '../data/399300.csv'
        data = pd.read_csv(data_path, encoding='GBK')

        data = data['涨跌幅'][-2::-1].values.astype(np.float32)

        train_data = dict()
        train_data['features'] = np.reshape(data[:3300], (33, self.num_steps, 1))
        train_data['labels'] = np.reshape(data[1:3301], (33, self.num_steps, 1))

        test_data = dict()
        test_data['features'] = np.reshape(data[3323:-1], (self.batch_size, self.num_steps, 1))
        test_data['labels'] = np.reshape(data[3324:], (self.batch_size, self.num_steps, 1))

        train_dataset = tf.contrib.data.Dataset.from_tensor_slices(train_data)
        test_dataset = tf.contrib.data.Dataset.from_tensor_slices(test_data).batch(self.batch_size)

        train_dataset = train_dataset.shuffle(10).batch(self.batch_size).repeat()

        return train_dataset, test_dataset
