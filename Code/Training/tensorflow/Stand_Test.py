''' Tiny version Test '''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils

path = '/home/mike/Documents/pyCharm/Location/tensorflow/model'
ckpt = tf.train.get_checkpoint_state(path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

def predict(test_data, test_label):
    with tf.Session() as sess:
        saver.restore(sess, ckpt.model_checkpoint_path)

        x = sess.graph.get_tensor_by_name('Placeholder:0')
        y = sess.graph.get_tensor_by_name('Placeholder_1:0')
        accuracy = sess.graph.get_tensor_by_name('Mean_1:0')

        print ("Testing Accuracy:", sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))

def main():
    test_csv = pd.read_csv('location25_test_v1.csv')
    test = np.array(test_csv)
    test_data = -np.hsplit(test, (0, 25))[1]
    test_label = np.hsplit(test, (0, 25))[2]
    test_label_OneHot = np_utils.to_categorical(test_label)
    predict(test_data, test_label_OneHot)

if (__name__ == '__main__'):
    main()
