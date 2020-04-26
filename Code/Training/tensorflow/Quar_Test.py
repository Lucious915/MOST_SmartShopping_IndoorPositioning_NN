''' Final version Test '''

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

        x = sess.graph.get_tensor_by_name('Input_layer/Placeholder:0')
        train_flag = sess.graph.get_tensor_by_name('Input_layer/Placeholder_2:0')
        keep_prob =  sess.graph.get_tensor_by_name('Input_layer/Placeholder_3:0')
        out = sess.graph.get_tensor_by_name('Output_layer/Relu:0')
        single_result = sess.run(out, feed_dict = {x: test_data[0, :].reshape(1, -1), train_flag: False, keep_prob: 1.0})
        all_result = sess.run(out, feed_dict = {x: test_data[:], train_flag: False, keep_prob: 1.0})

        location = np.floor_divide(np.argmax(single_result, 1), 4)
        correct_pred = np.equal(np.floor_divide(np.argmax(all_result, 1), 4), test_label.T)
        arruracy = np.mean(correct_pred.astype(int))

        print("Testing Location:", location)
        print ("Testing Accuracy:", arruracy)

def main():
    test_csv = pd.read_csv('location25_test_v1.csv')
    test = np.array(test_csv)
    test_data = -np.hsplit(test, (0, 25))[1]
    test_label = np.hsplit(test, (0, 25))[2]
    predict(test_data, test_label)

if (__name__ == '__main__'):
    main()
