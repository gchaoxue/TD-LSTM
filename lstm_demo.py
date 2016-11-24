import tensorflow as tf 
import numpy as np 

lstm_size = 6
num_class = 10
input_x = np.array([ [[0.2, 0.4, 0.7],[0, 0, 0]], 
					 [[0.5, 0.6, 0.2],[0.7, 0.8, 0.2]], 
					 [[0.9, 1.0, 0.2],[0.2, 0, 0]],
					 [[1.3, 1.4, 0.2],[1.5, 1.6, 0.2]] ])

input_x = np.ndarray(shape=np.shape(input_x), buffer=input_x)
# print(input_x)

x = tf.placeholder(tf.float32, [None,2,3])
used = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
length = tf.reduce_sum(used, reduction_indices=1)
length = tf.cast(length, tf.int32)

w = tf.Variable(tf.random_normal([lstm_size, num_class]))
b = tf.Variable(tf.zeros([num_class]))

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
output, state = tf.nn.dynamic_rnn(lstm,
								  x, 
								  dtype=tf.float32,
								  sequence_length=length)

batch_size = tf.shape(output)[0]
max_length = tf.shape(output)[1]
out_size = int( output.get_shape()[2] )
index = tf.range(0, batch_size) * max_length + (length - 1)
flat = tf.reshape(output, [-1, out_size])
last = tf.gather(flat, index)

# output = tf.transpose(output, [1,0,2])

# y = tf.nn.softmax(tf.matmul(output, w) + b)

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	las = sess.run(last, feed_dict={x: input_x})
	out = sess.run(output, feed_dict={x: input_x})
	print(out)
	print('++++++++++++++++++++++++++++')
	print(np.shape(las))
	print(las)

	# print(output)