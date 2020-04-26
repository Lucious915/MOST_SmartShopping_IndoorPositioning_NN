''' Tiny version Predict '''

import sys
import numpy as np
import tensorflow as tf

threshold = 30
saver = tf.train.import_meta_graph('/home/pc/Documents/Location/model/weights_best.ckpt.meta')

def predict(rssi_data_x):
    with tf.Session() as sess:
        saver.restore(sess, '/home/pc/Documents/Location/model/weights_best.ckpt')

        x = sess.graph.get_tensor_by_name('Input_layer/Placeholder:0')
        train_flag = sess.graph.get_tensor_by_name('Input_layer/Placeholder_2:0')
        out = sess.graph.get_tensor_by_name('Output_layer/Relu:0')
        predict = tf.argmax(out, 1)
        location = sess.run(predict, feed_dict = {x: rssi_data_x, train_flag: False})

    return int(str(location)[1:-1])

def main():
    last_rssi = sys.argv[2].split('-')[1:]
    current_rssi = sys.argv[3].split('-')[1:]
    last_rssi_np = np.array(last_rssi, dtype = np.float32, ndmin = 2)
    current_rssi_np = np.array(current_rssi, dtype = np.float32, ndmin = 2)

    last_local = int(sys.argv[1])
    current_local = predict(current_rssi_np)
    difference = np.linalg.norm(last_rssi_np - current_rssi_np)
    if ((last_local == -1) or (difference >= threshold)):
        print(current_local)
    else:
        if ((current_local % 5) == (last_local % 5)):
            print(current_local)
        else:
            print(last_local)

if (__name__ == '__main__'):
    main()
