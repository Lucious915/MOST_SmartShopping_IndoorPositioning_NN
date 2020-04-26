''' Tiny version Retrain '''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils

# hyper parameters
learning_rate = 0.0001
train_epochs = 1000000
display_epochs = 100
batch_size = 300

path = '/home/mike/Documents/Location/tensorflow/model'
ckpt = tf.train.get_checkpoint_state(path)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')

class NetWork():
    def __init__(self):
        pass

    def retrain(self, train_data, train_label):
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            self.x = sess.graph.get_tensor_by_name('Placeholder:0')
            self.y = sess.graph.get_tensor_by_name('Placeholder_1:0')
            self.out = sess.graph.get_tensor_by_name('Relu_2:0')

            accuracy = sess.graph.get_tensor_by_name('Mean_1:0')
            tf.summary.scalar(name = 'accuracy', tensor = accuracy)
            cost = sess.graph.get_tensor_by_name('add:0')
            tf.summary.scalar(name = 'cost', tensor = cost)
            optimizer = sess.graph.get_operation_by_name('Adam')

            writter = tf.summary.FileWriter('./logs/retrain', sess.graph)
            merged = tf.summary.merge_all()

            print ("\n-------------------------------Retraining Start---------------------------------\n")
            min_loss = 100
            batch_len = int(train_data.shape[0] / batch_size)

            for epoch in range(train_epochs):
                shuffle_idx = np.random.permutation(train_data.shape[0])
                for batch_idx in range(batch_len):
                    train_idx = shuffle_idx[(batch_idx * batch_size):(batch_idx * batch_size + batch_size)]
                    batch_xs = np.zeros(shape = (1, train_data.shape[1]))
                    batch_ys = np.zeros(shape = (1, train_label.shape[1]))
                    for idx in train_idx:
                         batch_xs = np.concatenate((batch_xs, train_data[idx][np.newaxis, :]), axis = 0)
                         batch_ys = np.concatenate((batch_ys, train_label[idx][np.newaxis, :]), axis = 0)

                    sess.run(optimizer, feed_dict = {self.x: batch_xs[1:], self.y: batch_ys[1:]})

                summary, loss, train_acc = sess.run([merged, cost, accuracy], feed_dict = {self.x: train_data, self.y: train_label})
                writter.add_summary(summary, epoch)

                if ((epoch + 1) % display_epochs == 0):
                    print("\nEpoch %02d" % (epoch + 1), ":", " Loss =", "{:.9f}".format(loss), " Accuracy =", train_acc)

                if (min_loss > loss):
                    min_loss = loss
                    saver.save(sess, './model/weights_min_loss.ckpt', global_step = epoch + 1)

def main():
    train_csv = pd.read_csv('location25_train_v1.csv')
    train = np.array(train_csv)
    train_data = -np.hsplit(train, (0, 25))[1]
    train_label = np.hsplit(train, (0, 25))[2]
    train_label_OneHot = np_utils.to_categorical(train_label)
    network = NetWork()
    network.retrain(train_data, train_label_OneHot)

if (__name__ == '__main__'):
    main()