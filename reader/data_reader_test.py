from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from reader import data_producer


class DataReaderTest(tf.test.TestCase):
    def setUp(self):
        self._series_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]

    def testDataProducer(self):
        raw_data = self._series_data
        batch_size = 3
        num_steps = 2
        x, y = data_producer.DataProducer(batch_size, num_steps).producer(raw_data)
        with self.test_session() as session:
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(session, coord=coord)
            try:
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
                self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
                xval, yval = session.run([x, y])
                self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
                self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
            finally:
                coord.request_stop()
                coord.join()


if __name__ == "__main__":
    tf.test.main()
