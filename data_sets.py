import tensorflow as tf
import numpy as np 

TEST_DATA_PATH = 'data/acl-14-short-data/test_slice_0.raw'
TRAIN_DATA_PATH = 'data/acl-14-short-data/train.raw'
WORD_EMB_PATH = 'data/embedding-results/sswe-h_slice.txt'

lstm_size = 128

def read_word_vector(path):
	word_vector = {}
	print('loading file:', path, '...')
	with open(path, 'r') as file:
		counter = 0
		while True:
			line = file.readline()
			if line == '':
				break
			counter = counter + 1
			print('loading word:', counter)
			line = line.split()
			word = line[0]
			print('word:', word)
			vector = [float(x) for x in line[1:]]
			word_vector[word] = vector
	return word_vector

def read_sentence_data(path):
	test_data = []
	print('loading file:', path, '...')
	with open(path, 'r') as file:
		counter = 0
		while True:
			text = file.readline()
			if text == '':
				break
			counter = counter + 1
			text = text.split()
			targets = file.readline().split()
			label = int(file.readline())
			'''
			print('test case:', counter)
			print(text)
			print(targets)
			print(label)
			'''
			test_data.append({'text': text,
							  'targets': targets,
							  'label': label})
	return test_data

word_vector = read_word_vector(WORD_EMB_PATH)
test_data = read_sentence_data(TEST_DATA_PATH)

print(word_vector)
'''
t = test_data[4]['text']
print(t)
print(t.index('$T$'))
'''
'''
print('length of the dict: ', len(wordVector))
for key in wordVector:
	print(key)
	print(wordVector[key])
'''