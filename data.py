import tensorflow as tf 
import numpy as np 

class Data:
	word_vec_dir = None
	train_data_dir = None
	test_data_dir = None

	def __init__(self,
				 word_vec_dir='data/embedding-results/sswe-h.txt',
				 train_data_dir='data/acl-14-short-data/train.raw',
				 test_data_dir='data/acl-14-short-data/test.raw'):
		self.word_vec_dir = word_vec_dir
		self.train_data_dir = train_data_dir
		self.test_data_dir = test_data_dir

	def get_default_test_dir(self):
		return 'data/acl-14-short-data/test_slice_0.raw'

	def get_train_data_dir(self):
		return self.train_data_dir

	def get_test_data_dir(self):
		return self.test_data_dir

	def read_word_vector(self):
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

		return word_vector

	def read_text_data(self, path):
		text_data = {'texts': [],
					 'targets': [],
					 'labels': []}
		with open(path, 'r') as file:
			cnt = 0
			while True:
				text = file.readline()
				if text == '':
					break
				cnt += 1
				
				text = text.split()
				targets = file.readline().split()
				label = int(file.readline())

				text_data['texts'].append(text)
				text_data['targets'].append(targets)
				text_data['labels'].append(label)

		return text_data