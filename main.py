'''
data format: [{'state': state_1, 'action': action_1}, ..., {'state': state_n, 'action': action_n}]
# version 1
network input: raw board state
network output: probability distribution
'''
import numpy as np
import copy, os
import random
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

BOARD_SIZE = 20
TRAINING_RATIO = 0.8
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE

def change_color(curr_color):
	if curr_color == 1:
		color = 2
	elif curr_color == 2:
		color = 1
	else:
		color = -1
	return color

def get_winner(file_dir):
	step_cnt = int(file_dir.split('-')[1].split('(')[0])

	if step_cnt % 2 == 1:
		winner = 1
	else:
		winner = 2

	return winner

def get_openning_steps_cnt(file_dir):
	think_time_limit = 100

	file = open(file_dir, 'r')
	file.readline()
	
	openning_steps_cnt = 0
	prev_think_time = 0

	for line in file.readlines():
		curr_think_time = int(line.split(',')[2])
		if abs(curr_think_time - prev_think_time) > think_time_limit:
			break
		prev_think_time = curr_think_time
		openning_steps_cnt += 1
	file.close()

	return openning_steps_cnt

def is_odd(num):
	return (num % 2 == 1)

def is_even(num):
	return (num % 2 == 0)

def print_matirx(matrix):
	row_cnt = len(matrix)
	col_cnt = len(matrix[0])
	for i in range(row_cnt):
		for j in range(col_cnt):
			print matrix[i][j],
		print '\n'
	print '\n'

# convert the psq file into a series of board state-action pairs
def conversion(file_dir):
	# get necessary variables: winner, openning_stpes_cnt
	winner = get_winner(file_dir)
	openning_steps_cnt = get_openning_steps_cnt(file_dir)

	# initialize the index of valid content
	st_cont_idx = openning_steps_cnt
	if (is_odd(st_cont_idx) and is_even(winner)) or (is_even(st_cont_idx) and is_odd(winner)):
		st_cont_idx -= 1

	# read file content
	file = open(file_dir, 'r')
	lines = file.readlines()[1:][:-1]
	game = lines[:-2]
	players = lines[-2:]
	
	game_length = len(game)

	# initialize the matrix with openning steps
	# matrix = [[0 for col in range(BOARD_SIZE)] for row in range(BOARD_SIZE)]
	matrix = np.zeros((BOARD_SIZE, BOARD_SIZE))
	color = 1
	for i in range(st_cont_idx):
		game_step = game[i].split(',')
		col, row = int(game_step[0]) - 1, int(game_step[1]) - 1
		matrix[row][col] = color
		color = change_color(color)

	# get state-action pair list
	state_action_pair_list = []
	for i in range(st_cont_idx, game_length, 2):
		game_step = game[i].split(',')
		if (i + 1) >= game_length:
			break
		next_game_step = game[i + 1].split(',')

		curr_col, curr_row = int(game_step[0]) - 1, int(game_step[1]) - 1
		matrix[curr_row][curr_col] = color
		color = change_color(color)

		next_col, next_row = int(next_game_step[0]) - 1, int(next_game_step[1]) - 1
		next_move_id = next_row * BOARD_SIZE + next_col

		curr_board_state = copy.deepcopy(matrix) # i don't know why deepcopy is actually shallow copy...
		state_action_pair = {'state': curr_board_state, 'action': next_move_id}
		state_action_pair_list.append(state_action_pair)

		matrix[next_row][next_col] = color
		color = change_color(color)

	file.close()

	return state_action_pair_list

def read_file_in_folder(folder_dir):
	state_action_pair_list = []
	for file in os.listdir(folder_dir):
		file_dir = os.path.join(folder_dir, file)
		if os.path.splitext(file_dir)[1] == '.psq':
			# print 'reading... ', file
			curr_state_action_pair_list = conversion(file_dir)
			state_action_pair_list.extend(curr_state_action_pair_list)
	return state_action_pair_list

# get (x_train, y_train) from training_dataset or (x_test, y_test) from testing dataset
def get_x_y_dataset(data):
	x_t, y_t = [], []
	for d in data:
		action = d['action']
		state = d['state']
		x_t.append(state)
		y_t.append(action)
	x_t = np.array(x_t)
	y_t = np.array(y_t)
	return x_t, y_t

# partition the dataset into (x_train, y_train), (x_test, y_test)
def partition_dataset(data_list):
	data_cnt = len(data_list)
	training_data_cnt = int(data_cnt * TRAINING_RATIO)

	training_data = data_list[:training_data_cnt]
	testing_data = data_list[training_data_cnt:]

	x_train, y_train = get_x_y_dataset(training_data)
	x_test, y_test = get_x_y_dataset(testing_data)

	return x_train, y_train, x_test, y_test

def next_move_id_pos_conversion(next_move_id):
	row = next_move_id / BOARD_SIZE
	col = next_move_id % BOARD_SIZE
	return row, col

def next_move_pos_id_conversion(next_move_row, next_move_col):
	next_move_id = next_move_row * BOARD_SIZE + next_move_col
	return next_move_id

def rotate_90(board, next_move):
	board_rotate_90 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_90[j][n - 1 - i] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_90 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_90, next_move_rotate_90

def rotate_180(board, next_move):
	board_rotate_180 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_180[n - 1 - i][n - 1 - j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_180 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_180, next_move_rotate_180

def rotate_270(board, next_move):
	board_rotate_270 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_270[n - 1 - j][i] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_270 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_270, next_move_rotate_270

def mirror_y_axis(board, next_move):
	board_mirror_y = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_mirror_y[i][n - 1 - j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_mirror_y = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_mirror_y, next_move_mirror_y

def mirror_x_axis(board, next_move):
	board_mirror_x = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_mirror_x[n - 1 - i][j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_mirror_x = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_mirror_x, next_move_mirror_x

# rotate the board in 4 directions and mirror the board in 2 directions to get additional 8 transformations
def rotate_and_mirror(x_train, y_train):
	appended_x_train, appended_y_train = [], []

	num_of_training = len(y_train)
	for i in range(num_of_training):
		print 'data '+str(i)
		# type 1: original with mirroring on x and y axis
		# 0. original board
		#print 'original...'
		board, next_move = x_train[i], y_train[i]
		appended_x_train.append(board)
		appended_y_train.append(next_move)
		# 1. mirror on x-axis
		#print 'mirror on x-axis...'
		board_mirror_x, next_move_mirror_x = mirror_x_axis(board, next_move)
		appended_x_train.append(board_mirror_x)
		appended_y_train.append(next_move_mirror_x)
		# 2. mirror on y-axis
		#print 'mirror on y-axis...'
		board_mirror_y, next_move_mirror_y = mirror_y_axis(board, next_move)
		appended_x_train.append(board_mirror_y)
		appended_y_train.append(next_move_mirror_y)

		# type 2: 90 degrees rotation with mirroring on x and y axis
		# 3. roate 90 degrees
		#print 'rotate 90 degrees...'
		board_rotate_90, next_move_rotate_90 = rotate_90(board, next_move)
		appended_x_train.append(board_rotate_90)
		appended_y_train.append(next_move_rotate_90)
		# 4. mirror on x-axis after rotating 90 degrees
		#print 'rotate 90 degrees & mirror on x-axis...'
		board_rotate_90_mirror_x, next_move_rotate_90_mirror_x = mirror_x_axis(board_rotate_90, next_move_rotate_90)
		appended_x_train.append(board_rotate_90_mirror_x)
		appended_y_train.append(next_move_rotate_90_mirror_x)
		# 5. mirror on y-axis after rotating 90 degrees
		#print 'rotate 90 degrees & mirror on y-axis...'
		board_rotate_90_mirror_y, next_move_rotate_90_mirror_y = mirror_y_axis(board_rotate_90, next_move_rotate_90)
		appended_x_train.append(board_rotate_90_mirror_y)
		appended_y_train.append(next_move_rotate_90_mirror_y)

		# type 3: 180 degrees rotation with mirroring on x axis
		# 6. rotate 180 degrees
		#print 'rotate 180 degrees...'
		board_rotate_180, next_move_rotate_180 = rotate_180(board, next_move)
		appended_x_train.append(board_rotate_180)
		appended_y_train.append(next_move_rotate_180)
		# 7. mirror on x-axis after rotating 180 degrees
		#print 'rotate 180 degrees & mirror on x-axis...'
		board_rotate_180_mirror_x, next_move_roate_180_mirror_x = mirror_x_axis(board_rotate_180, next_move_rotate_180)
		appended_x_train.append(board_rotate_180_mirror_x)
		appended_y_train.append(next_move_roate_180_mirror_x)

		# type 4: 270 degrees rotation
		# 8. rotate 270 degrees 
		#print 'rotate 270 degrees...'
		board_rotate_270, next_move_rotate_270 = rotate_270(board, next_move)
		appended_x_train.append(board_rotate_270)
		appended_y_train.append(next_move_rotate_270)

	appended_x_train = np.array(appended_x_train)
	appended_y_train = np.array(appended_y_train)

	return appended_x_train, appended_y_train

def shuffle_data(data_list):
	return random.shuffle(data_list)

def sl_training(data_list):
	# randomly shuffle the data
	shuffle_data(data_list)

	# get training and testing data
	x_train, y_train, x_test, y_test = partition_dataset(data_list)
	
	# rotate and mirror the training data
	print 'rotating and mirroring...'
	rotate_and_mirror(x_train, y_train)

	# preprocessing
	x_train = x_train.reshape(x_train.shape[0], BOARD_SIZE, BOARD_SIZE, 1)
	x_test = x_test.reshape(x_test.shape[0], BOARD_SIZE, BOARD_SIZE, 1)
	input_shape = (BOARD_SIZE, BOARD_SIZE, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 2
	x_test /= 2
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
	y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

	# model specification
	model = Sequential()
	
	# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	# model.add(Conv2D(64, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Dropout(0.25))
	# model.add(Flatten())
	# model.add(Dense(128, activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(NUM_CLASSES, activation='softmax'))

	num_of_intermediate_layers = 12
	model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same', input_shape=input_shape))
	for i in range(num_of_intermediate_layers):
		model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
	model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
	model.add(Flatten())
	#model.add(Bias())
	model.add(keras.layers.core.Activation(activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	
	# training
	batch_size = 128
	epochs = 12
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test))
	
	# testing
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

if __name__ == '__main__':
	dataset_dir = '/home/teamhuang/gomoku/gomoku_dataset'
	dataset_folder_dir_list = []
	for file in os.listdir(dataset_dir):
		file_dir = os.path.join(dataset_dir, file)
		if os.path.isdir(file_dir):
			dataset_folder_dir_list.append(file_dir)
	
	data_list = []
	for dataset_folder_dir in dataset_folder_dir_list:
		curr_data_list = read_file_in_folder(dataset_folder_dir)
		data_list.extend(curr_data_list)

	sl_training(data_list)
	# file_dir = '/Users/panling/Desktop/gomoku_drl/gomoku_dataset/' + 'opening_1_380/' + '0x1-22(1).psq'
	# curr_list = conversion(file_dir)
	# for item in curr_list:
	# 	state = item['state']
	# 	move_id = item['action']

	# 	move_row = move_id / 20
	# 	move_col = move_id % 20
	# 	print 'move_id: ', move_id, ' (', move_row, move_col, ')'
	# 	state[move_row][move_col] = '*'

	# 	print_matirx(state)
