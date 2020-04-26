''' Original version Train'''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils

# hyper parameters
learning_rate = 0.001
train_epochs = 1000000
display_epochs = 100
batch_size = 300

# dropout rate
dropout_rate = 0.75

class NetWork():
    def __init__(self):
        pass

    def __conv_2d(self, name, input_ts, kernel_size, filters, act = tf.nn.relu,
                 kernel = tf.truncated_normal_initializer(stddev = 0.1),
                 bias = tf.constant_initializer(value = 0.1)):
        return tf.layers.conv2d(input_ts, filters, kernel_size, padding = 'valid', activation = act,
                                kernel_initializer = kernel, bias_initializer = bias, name = name)

    def __fc_flatten(self, name, input_ts):
        return tf.layers.flatten(input_ts, name=name)

    def __fc_dense(self, name, input_ts, units, act = tf.nn.relu,
                  kernel = tf.truncated_normal_initializer(stddev=0.1),
                  bias = tf.constant_initializer(value=0.1)):
        return tf.layers.dense(input_ts, units, activation = act,
                                kernel_initializer = kernel, bias_initializer = bias, name = name)

    def __batch_norm(self, name, input_ts, train_flag):
        return tf.contrib.layers.batch_norm(input_ts, decay = 0.9, center = True, scale = True, epsilon = 1e-3,
                                             is_training = train_flag, scope = name)

    def __drop_out(self, name, input_ts, keep_prob):
        return tf.nn.dropout(input_ts, keep_prob, name = name)
 

    def creat_network(self):
        train_input = 25
        train_classes = 120
        test_classes = 22

        with tf.name_scope('Input_layer'):
            self.x = tf.placeholder(tf.float32, [None, train_input])
            self.y = tf.placeholder(tf.float32, [None, train_classes])
            self.valid_y = tf.placeholder(tf.float32, [None, test_classes])
            self.keep_prob = tf.placeholder(tf.float32)
            self.train_flag = tf.placeholder(tf.bool)

        x_ = tf.reshape(self.x, shape = [-1, 5, 5, 1])
        conv = self.__conv_2d('Conv_layer', x_, 2, 120)
        fc_flat = self.__fc_flatten('Flat_layer', conv)
        fc = self.__fc_dense('FC_layer', fc_flat, 240)
        batch = self.__batch_norm('Batch_norm', fc, self.train_flag)
        drop = self.__drop_out('Drop_layer', batch, self.keep_prob)
        self.out = self.__fc_dense('Output_layer', drop, 120)

        self.lossL2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name]) * 0.001

    def train(self, train_data, train_label, test_data, test_label):
        with tf.Session() as sess:
            with tf.name_scope('Cost'):
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.y)) + self.lossL2
                tf.summary.scalar(name = 'cost', tensor = cost)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

            with tf.name_scope('Accuracy'):
                correct_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                tf.summary.scalar(name = 'accuracy', tensor = accuracy)

            with tf.name_scope('Validation'):
                validation_pred = tf.equal(tf.cast(tf.div(tf.argmax(self.out, 1), 4), tf.int64), tf.argmax(self.valid_y, 1))
                validation = tf.reduce_mean(tf.cast(validation_pred, tf.float32))
                tf.summary.scalar(name = 'validation', tensor = validation)

            writter = tf.summary.FileWriter('./logs/train', sess.graph)
            merged = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            saver1=tf.train.Saver(max_to_keep = 1)
            saver2=tf.train.Saver(max_to_keep = 1)

            print ("\n--------------------------------Training Start----------------------------------\n")
            max_acc = 0
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

                    sess.run(optimizer, feed_dict = {self.x: batch_xs[1:], self.y: batch_ys[1:], self.keep_prob: dropout_rate, self.train_flag: True})

                summary1, loss, train_acc = sess.run([merged, cost, accuracy],
                                                     feed_dict = {self.x: train_data, self.y: train_label, self.valid_y: np.zeros(shape = (1, 22)), self.keep_prob: 1.0, self.train_flag: False})
                summary2, valid_acc = sess.run([merged, validation],
                                                     feed_dict = {self.x: test_data, self.y: np.zeros(shape = (80, 120)), self.valid_y: test_label, self.keep_prob: 1.0, self.train_flag: False})

                writter.add_summary(summary1, epoch)
                writter.add_summary(summary2, epoch)

                if ((epoch + 1) % display_epochs == 0):
                    print("\nEpoch %02d" % (epoch + 1), ":", " Loss =", "{:.9f}".format(loss), " Train_ACC =", train_acc, " Valid_ACC =", valid_acc)

                if (min_loss > loss):
                    min_loss = loss
                    saver1.save(sess, './model/weights_min_loss.ckpt', global_step = epoch + 1)
                if (max_acc < valid_acc):
                    max_acc = valid_acc
                    saver2.save(sess, './model/weights_max_acc.ckpt', global_step = epoch + 1)

def main():
    train_csv = pd.read_csv('location25_train_v2.csv')
    test_csv = pd.read_csv('location25_test_v2.csv')
    train = np.array(train_csv)
    test = np.array(test_csv)
    train_data = -np.hsplit(train, (0, 25))[1]
    train_label = np.hsplit(train, (0, 25))[2]
    train_label_OneHot = np_utils.to_categorical(train_label)
    test_data = -np.hsplit(test, (0, 25))[1]
    test_label = np.hsplit(test, (0, 25))[2]
    test_label_OneHot = np_utils.to_categorical(test_label)
    network = NetWork()
    network.creat_network()
    network.train(train_data, train_label_OneHot, test_data, test_label_OneHot)

if (__name__ == '__main__'):
    main()