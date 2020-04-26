import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils import np_utils

# Training Parameters
learning_rate = 0.001
training_epochs = 100000
batch_size = 128
display_epochs = 100

# Network Parameters
num_input = 4 # MNIST data input (img shape: 28*28)
time_seqlen = 10 # time sequence length
num_hidden = 128 # hidden layer num of features
num_classes = 16 # MNIST total classes (0-9 digits)

csv_filename = '07-12TR_merge.csv'
model_graph_path = './model_tr_rnn_0718'
log_path = './logs_tr_rnn_0718/train'
ckpt_path = './model_tr_rnn_0718/weights_final.ckpt'

class NetWork():

	def createRNN(self):

		# tf Graph input
		with tf.name_scope('Input_layer'):
			self.X = tf.placeholder(tf.float32, [None, time_seqlen, num_input])
			self.Y = tf.placeholder(tf.float32, [None, num_classes])

		# Define weights
		self.weights = {'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))}
		self.biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

		# Prepare data shape to match `rnn` function requirements
		# Current data input shape: (batch_size, time_seqlen, n_input)
		# Required shape: 'time_seqlen' tensors list of shape (batch_size, n_input)

		# Unstack to get a list of 'time_seqlen' tensors of shape (batch_size, n_input)
		x = tf.unstack(self.X, time_seqlen, 1)

		# Define a lstm cell with tensorflow
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

		# Get lstm cell output
		self.outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

		
		
		


	def train(self, train_data, train_label):

		# Linear activation, using rnn inner loop last output
		with tf.name_scope('Output_layer'):
			logits = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
			prediction = tf.nn.softmax(logits)

		
		with tf.Session() as sess:
			tf.train.write_graph(sess.graph, model_graph_path, 'graph.pbtxt')

		# Define loss and optimizer
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=logits, labels=self.Y))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(loss_op)

		# Evaluate model (with test logits, for dropout to be disabled)
		correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initialize the variables (i.e. assign their default value)
		init = tf.global_variables_initializer()

		
		# Start training
		with tf.Session() as sess:
			writter = tf.summary.FileWriter(log_path, sess.graph)
			merged = tf.summary.merge_all()
			saver = tf.train.Saver(max_to_keep = 1)
			min_loss = 2
			# Run the initializer
			sess.run(init)
			batch_len = int(np.array(train_data).shape[0] / batch_size)

			print ("\n--------------------------------Training Start----------------------------------\n")
			for epoch in range(training_epochs):
				shuffle_idx = np.random.permutation(np.array(train_data).shape[0])
				for batch_idx in range(batch_len):
					train_idx = shuffle_idx[(batch_idx * batch_size):(batch_idx * batch_size + batch_size)]
					start_concatenate = 1
					for idx in train_idx:
						if(start_concatenate == 1):
							batch_xs = train_data[idx][np.newaxis,:, :]
							batch_ys = train_label[idx][np.newaxis, :]
							start_concatenate = 0
						else:
							batch_xs = np.concatenate((batch_xs, train_data[idx][np.newaxis,:, :]), axis = 0)
							batch_ys = np.concatenate((batch_ys, train_label[idx][np.newaxis, :]), axis = 0)
					sess.run(train_op, feed_dict={self.X: batch_xs, self.Y: batch_ys})

				loss, acc = sess.run( [loss_op, accuracy], feed_dict={self.X: train_data, self.Y: train_label})
				# writter.add_summary(summary, epoch)

				if ((epoch) % display_epochs == 0):
					print("Step " + str(epoch) + ", Loss= " + \
						  "{:.9f}".format(loss) + ", Training Accuracy= " + \
						  "{:.9f}".format(acc))

				if (min_loss > loss):
					min_loss = loss
					saver.save(sess, ckpt_path, global_step = epoch + 1)



def main():
	train_csv = pd.read_csv(csv_filename)
	train = np.array(train_csv)
	train_split = -np.hsplit(train, (1, 13))[1]
	print train_split
	train_data = np.concatenate([np.hsplit(train_split,12)[0],np.hsplit(train_split,12)[5],np.hsplit(train_split,12)[6],np.hsplit(train_split,12)[11]],axis=1)
	print train_data
	train_label = np.hsplit(train, (0, 28))[2]
	print train_label
	train_data_seq = []
	train_label_seq = []
	for i in range(0,train_label.size-time_seqlen):
		if(train_label[i]!=train_label[i+time_seqlen]):
			continue
		train_data_seq.append(train_data[i:i+time_seqlen])
		train_label_seq.append(train_label[i])


	train_label_OneHot = np_utils.to_categorical(train_label_seq)
	print train_label_OneHot
	network = NetWork()
	network.createRNN()
	network.train(train_data_seq, train_label_OneHot)

if (__name__ == '__main__'):
	main()
