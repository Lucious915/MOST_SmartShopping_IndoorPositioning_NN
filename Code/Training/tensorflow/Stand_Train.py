''' Tiny version Train'''

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils

# hyper parameters
learning_rate = 0.001
train_epochs = 1000000
display_epochs = 100
batch_size = 300

class NetWork():
    def __init__(self):
        pass

    def creat_network(self):
        self.x = tf.placeholder(tf.float32, [None, 25])
        self.y = tf.placeholder(tf.float32, [None, 30])
        self.keep_prob = tf.placeholder(tf.float32)
        x_ = tf.reshape(self.x, shape = [-1, 5, 5, 1])
        x = (x_ - 40) / 80

        convw = tf.Variable(tf.truncated_normal([3, 3, 1, 15], dtype = tf.float32, stddev = 0.1), name = 'conv_weight')
        convb = tf.Variable(tf.constant(0.1, shape = [15], dtype = tf.float32), trainable = True, name = 'conv_bias')
        conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, convw, [1, 1, 1, 1], padding = 'VALID'), convb))

        shape = int(np.prod(conv.get_shape()[1:]))
        flat = tf.reshape(conv, shape = [-1, shape])
        fc1w = tf.Variable(tf.truncated_normal([shape, 10], dtype = tf.float32, stddev = 0.1), name = 'fc1_weight')
        fc1b = tf.Variable(tf.constant(0.1, shape = [10], dtype = tf.float32), trainable = True, name = 'fc1_bias')
        fc1o = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat, fc1w), fc1b))
        fc2w = tf.Variable(tf.truncated_normal([10, 30], dtype = tf.float32, stddev = 0.1), name = 'fc2_weight')
        fc2b = tf.Variable(tf.constant(0.1, shape = [30], dtype = tf.float32), trainable = True, name = 'fc2_bias')
        self.out = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1o, fc2w), fc2b))

        self.lossL2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name]) * 0.0005

    def train(self, train_data, train_label):
        with tf.Session() as sess:
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.y)) + self.lossL2
            tf.summary.scalar(name = 'cost', tensor = cost)
            optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

            correct_pred = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar(name = 'accuracy', tensor = accuracy)

            writter = tf.summary.FileWriter('./logs/train', sess.graph)
            merged = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            saver=tf.train.Saver(max_to_keep = 1)

            print ("\n--------------------------------Training Start----------------------------------\n")
            min_loss = 2
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
    network.creat_network()
    network.train(train_data, train_label_OneHot)

if (__name__ == '__main__'):
    main()
