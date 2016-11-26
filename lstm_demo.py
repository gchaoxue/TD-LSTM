import tensorflow as tf 
import numpy as np 

from data import Data

data = Data()
# data.set_as_test()
data.get_data()

VECTOR_DIM = data.vector_dim
LSTM_SIZE = VECTOR_DIM
MAX_TIME_STEP = 50
NUM_CLASS = 3
LEARNING_RATE = 1
HIDDEN1_SIZE = 60
HIDDEN2_SIZE = 30

def mask_output(output, seq_length):
	# masking the output vector
	batch_size = tf.shape(output)[0]
	max_length = tf.shape(output)[1]
	out_size = int( output.get_shape()[2] )
	index = tf.range(0, batch_size) * max_length + (seq_length - 1)
	flat = tf.reshape(output, [-1, out_size])
	return tf.gather(flat, index)

left_input = tf.placeholder(tf.float32, [None,None,VECTOR_DIM])
left_seq_length = tf.placeholder(tf.int32, [None])

right_input = tf.placeholder(tf.float32, [None,None,VECTOR_DIM])
right_seq_length = tf.placeholder(tf.int32, [None])

y_ = tf.placeholder(tf.float32, [None, NUM_CLASS])

hidden1_weights = tf.Variable(tf.random_normal([LSTM_SIZE*2, HIDDEN1_SIZE]))
hidden1_biases = tf.Variable(tf.zeros([HIDDEN1_SIZE]))

hidden2_weights = tf.Variable(tf.random_normal([HIDDEN1_SIZE, HIDDEN2_SIZE]))
hidden2_biases = tf.Variable(tf.zeros([HIDDEN2_SIZE]))

weights = tf.Variable(tf.random_normal([HIDDEN2_SIZE, NUM_CLASS]))
biases = tf.Variable(tf.random_normal([NUM_CLASS]))

with tf.variable_scope('left_lstm'):
	left_lstm = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE)
	left_outputs, left_states = tf.nn.dynamic_rnn(left_lstm,
												  left_input,
												  dtype=tf.float32,
												  sequence_length=left_seq_length)

with tf.variable_scope('right_lstm'):
	right_lstm = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE)
	right_outputs, right_states = tf.nn.dynamic_rnn(right_lstm,
										  			right_input,
										  			dtype=tf.float32,
										  			sequence_length=right_seq_length)

left_outputs = mask_output(left_outputs, left_seq_length)
right_outputs = mask_output(right_outputs, right_seq_length)

comb_outputs = tf.concat(1, [left_outputs, right_outputs])

hidden1_layer = tf.sigmoid(tf.matmul(comb_outputs, hidden1_weights)+hidden1_biases)
hidden2_layer = tf.sigmoid(tf.matmul(hidden1_layer, hidden2_weights)+hidden2_biases)

y = tf.nn.softmax(tf.sigmoid(tf.matmul(hidden2_layer, weights) + biases))

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

true_count = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
acc = tf.reduce_mean(tf.cast(true_count, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	for _ in range(1000):
		data_batch = data.get_next_batch(data.train, 200)
		batch_feed_dict = {
			left_input: data_batch['left_input'],
			right_input: data_batch['right_input'],
			left_seq_length: data_batch['left_seq_length'],
			right_seq_length: data_batch['right_seq_length'],
			y_: data_batch['labels']
		}
		sess.run(train_step, feed_dict=batch_feed_dict)
		acc_val = sess.run(acc, feed_dict=batch_feed_dict)
		print('step: %.4d, learning rate: %.2f, acc: %.3f' % (_, LEARNING_RATE, acc_val))

	# print(output)
