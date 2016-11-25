import tensorflow as tf 
import numpy as np 

class Data:
	'''Data reading and preparing
	- Attr - word_vec_dir: direction of word embedding file
	- Attr - train_data_dir: direction of traing data
	- Attr - test_data_dir: direction of test data

	- Attr - 
	'''
	word_vec_dir = None
	train_data_dir = None
	test_data_dir = None

	vector_dim = None
	max_length_left = None
	max_length_right = None
	seq_length_left = None
	seq_length_right = None

	train = None
	test = None

	def __init__(self,
				 word_vec_dir='data/embedding-results/sswe-h.txt',
				 train_data_dir='data/acl-14-short-data/train.raw',
				 test_data_dir='data/acl-14-short-data/test.raw'):
		self.word_vec_dir = word_vec_dir
		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir

	# reading small data-set for testing
	def set_as_test(self):
		self.test_data_dir = 'data/acl-14-short-data/test-test.raw'
		self.train_data_dir = 'data/acl-14-short-data/train-test.raw'
		self.word_vec_dir = 'data/embedding-results/sswe-h-test.txt'

	def get_train_data_dir(self):
		return self.train_data_dir

	def get_test_data_dir(self):
		return self.test_data_dir

	def get_word_vec_dir(self):
		return self.word_vec_dir

	def read_word_vector(self):
		# returning a dict {word: vector}
		word_vector = {}
		print('loading word-vector from ' + self.word_vec_dir + '...')
		with open(self.word_vec_dir, 'r') as file:
			cnt = 0
			while True:
				line = file.readline()
				# process till the EOF
				if line == '':
					break
				cnt += 1
				line = line.split()
				word = line[0]
				vector = [float(x) for x in line[1:]]
				word_vector[word] = vector
				if cnt == 1:
					self.vector_dim = len(vector)

		return word_vector

	def read_text_data(self,
					   data_type):
		'''
		returning text-data {'left_text': a word-list containing left-to-target text,
							 'right_text': a word-list containing target-to-right text,
							 'targets': a word-list containing target words,
							 'labels': label vector}
		'''
		path = self.test_data_dir
		if data_type == 'train':
			path = self.train_data_dir

		text_data = {'left_text': [],
					 'right_text': [],
					 'targets': [],
					 'labels': []}

		with open(path, 'r') as file:
			while True:
				text = file.readline()
				if text == '':
					break
				targets = file.readline()
				label = file.readline()
				
				target_index = text.index('$T$')
				left_text = text[0:target_index+3]
				right_text = text[target_index:]

				left_text = left_text.replace('$T$', targets)
				right_text = right_text.replace('$T$', targets)

				targets = targets.split()
				left_text = left_text.split()
				right_text = right_text.split()
				right_text.reverse()

				label = int(label)

				text_data['left_text'].append(left_text)
				text_data['right_text'].append(right_text)
				text_data['targets'].append(targets)
				text_data['labels'].append(label)

		return text_data

	def word_to_vec(self, text_data, word_vector):
		'''
			returning input vector with two directions
		'''
		left_input_vector = []
		right_input_vector = []

		for text in text_data['left_text']:
			input_vec = []
			for word in text:
				if word in word_vector:
					input_vec.append(word_vector[word])
				else:
					input_vec.append(word_vector['<unk>'])
			left_input_vector.append(input_vec)


		print(len(max(left_input_vector)))

		max_length_left = len(max(left_input_vector))
		print('MAX_LENGTH ' + str(max_length_left))

		self.max_length_left = max_length_left
		for seq in left_input_vector:
			for i in range(len(seq), max_length_left):
				seq.append([float(0) for _ in range(self.vector_dim)])

		for text in text_data['right_text']:
			input_vec = []
			for word in text:
				if word in word_vector:
					input_vec.append(word_vector[word])
				else:
					# '<unk>' for undefined words in word embedding
					input_vec.append(word_vector['<unk>'])
			right_input_vector.append(input_vec)


		max_length_right = len(max(left_input_vector))
		self.max_length_right = max_length_right
		for seq in right_input_vector:
			for i in range(len(seq), max_length_right):
				seq.append([float(0) for _ in range(self.vector_dim)])

		for seq in left_input_vector:
			print(len(seq))

		print('---------------------------------')

		for seq in right_input_vector:
			print(len(seq))

		return {'left': np.array(left_input_vector, dtype=float),
				'right': np.array(right_input_vector, dtype=float)}


	def get_data(self):
		word_vector = self.read_word_vector()
		self.test = self.word_to_vec(self.read_text_data(data_type='test'),
										  word_vector)
		self.train = self.word_to_vec(self.read_text_data(data_type='train'),
										   word_vector)
