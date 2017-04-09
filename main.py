import numpy as np
import copy, os
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
from keras import backend as K
from rotation_and_reflection import *

BOARD_SIZE = 20
EMPTY = 0
TRAINING_RATIO = 0.8
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE
FEATURES_MAP = {
				'connect_five': ['xxxxx'],
				'alive_four': ['_xxxx_'],
				'sleep_four': ['_xxxx', 'x_xxx', 'xx_xx', 'xxx_x', 'xxxx_'], 

				'alive_three': ['__xxx_', '_x_xx_', '_xx_x_', '_xxx__'],
				'sleep_three': ['__xxx', '_x_xx', '_xx_x', 'x__xx', '_xxx_', 'x_x_x', 'xx__x', 'x_xx_','xx_x_', 'xxx__'],

				'alive_two': ['___xx_', '__x_x_', '_x__x_', '__xx__', '_x_x__', '_xx___'],
				'sleep_two': ['___xx', '__x_x', '__xx_', '_x__x', '_x_x_', 'x___x', 'x__x_', '_xx__', 'x_x__', 'xx___']
			   }
FEATURES_VALUE_MAP = {'connect_five': 7, 'alive_four': 6, 'sleep_four': 5, 'alive_three': 4, 'sleep_three': 3, 'alive_two': 2, 'sleep_two': 1}
PADDING_SIZE = 4

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

def print_np_matirx(matrix):
	row_cnt, col_cnt = matrix.shape
	for i in range(row_cnt):
		for j in range(col_cnt):
			print int(matrix[i][j]), ' ',
		print '\n'
	print '\n'

# convert the psq file into a series of board state-action pairs and winners
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

		if (i + 2) >= game_length:
			transition_move_id = -1
		else:
			transition_game_step = game[i + 2].split(',')
			transition_col, transition_row = int(transition_game_step[0]) - 1, int(transition_game_step[1]) - 1
			transition_move_id = transition_row * BOARD_SIZE + transition_col

		curr_board_state = copy.deepcopy(matrix) # i don't know why deepcopy is actually shallow copy...
		state_action_pair = {'state': curr_board_state, 'action': next_move_id, 'winner': winner, 'transition_move': transition_move_id}
		state_action_pair_list.append(state_action_pair)

		matrix[next_row][next_col] = color
		color = change_color(color)

	file.close()

	return state_action_pair_list

def convert_game_to_feature_planes(data_list):
	for data in data_list:
		new_move_1_id = data['action']
		next_move_2_id = data['action_next']

def read_file_in_folder(folder_dir):
	state_action_pair_list = []
	for file in os.listdir(folder_dir):
		file_dir = os.path.join(folder_dir, file)
		if os.path.splitext(file_dir)[1] == '.psq':
			curr_state_action_pair_list = conversion(file_dir)
			state_action_pair_list.append(curr_state_action_pair_list)
	return state_action_pair_list

# connect-five:7 alive-four:6 sleep-four:5 alive-three:4 sleep-three:3 alive-two:2 sleep-two:1 none:0
def check_match_pattern(status):
	features_cnt = len(FEATURES_MAP)
	# connect-five
	for pattern in FEATURES_MAP['connect_five']:
		if pattern in status:
			return features_cnt
	# alive-four
	for pattern in FEATURES_MAP['alive_four']:
		if pattern in status:
			return (features_cnt - 1)
	# sleep-four
	for pattern in FEATURES_MAP['sleep_four']:
		if pattern in status:
			return (features_cnt - 2)
	# alive-three
	for pattern in FEATURES_MAP['alive_three']:
		if pattern in status:
			return (features_cnt - 3)
	# sleep-three
	for pattern in FEATURES_MAP['sleep_three']:
		if pattern in status:
			return (features_cnt - 4)
	# alive-two
	for pattern in FEATURES_MAP['alive_two']:
		if pattern in status:
			return (features_cnt - 5)
	# sleep-two
	for pattern in FEATURES_MAP['sleep_two']:
		if pattern in status:
			return (features_cnt - 6)
	return (features_cnt - 7)

def get_macro_feature_plane(board_state, player):
	macro_feature_plane = np.zeros((BOARD_SIZE, BOARD_SIZE))

	# padding the edege rows and columns with 0
	board_state = np.lib.pad(board_state, PADDING_SIZE, 'constant', constant_values=0)
	st, ed = PADDING_SIZE, BOARD_SIZE + PADDING_SIZE

	for i in range(st, ed):
		for j in range(st, ed):
			curr_pattern = get_pos_pattern(board_state, i, j, player)
			macro_feature_plane[i - PADDING_SIZE][j - PADDING_SIZE] = curr_pattern

	return macro_feature_plane

def get_pos_pattern(board_state, i, j, player):
	# !!! assuming that the board_state is properly padded !!!
	
	if board_state[i][j] == 0:
		return 0

	# horizontal
	horizontal_status_str = ''
	for k in range(j - PADDING_SIZE, j + (PADDING_SIZE + 1)):
		if board_state[i][k] == player:
			horizontal_status_str += 'x'
		elif board_state[i][k] == EMPTY:
			horizontal_status_str += '_'
		else:
			horizontal_status_str += 'o'
	horizontal_pattern = check_match_pattern(horizontal_status_str)
	# verticle
	verticle_status_str = ''
	for k in range(i - PADDING_SIZE, i + (PADDING_SIZE + 1)):
		if board_state[k][j] == player:
			verticle_status_str += 'x'
		elif board_state[k][j] == EMPTY:
			verticle_status_str += '_'
		else:
			verticle_status_str += 'o'
	verticle_pattern = check_match_pattern(verticle_status_str)
	# diagonal_1
	diagonal_1_status_str = ''
	for k in range(PADDING_SIZE, 0, -1):
		if board_state[i - k][j - k] == player:
			diagonal_1_status_str += 'x'
		elif board_state[i - k][j - k] == EMPTY:
			diagonal_1_status_str += '_'
		else:
			diagonal_1_status_str += 'o'
	for k in range(PADDING_SIZE + 1):
		if board_state[i + k][j + k] == player:
			diagonal_1_status_str += 'x'
		elif board_state[i + k][j + k] == EMPTY:
			diagonal_1_status_str += '_'
		else:
			diagonal_1_status_str += 'o'
	diagonal_1_pattern = check_match_pattern(diagonal_1_status_str)
	# diagonal_2
	diagonal_2_status_str = ''
	for k in range(PADDING_SIZE, 0, -1):
		if board_state[i + k][j - k] == player:
			diagonal_2_status_str += 'x'
		elif board_state[i + k][j - k] == EMPTY:
			diagonal_2_status_str += '_'
		else:
			diagonal_2_status_str += 'o'
	for k in range(PADDING_SIZE + 1):
		if board_state[i - k][j + k] == player:
			diagonal_2_status_str += 'x'
		elif board_state[i - k][j + k] == EMPTY:
			diagonal_2_status_str += '_'
		else:
			diagonal_2_status_str += 'o'
	diagonal_2_pattern = check_match_pattern(diagonal_2_status_str)

	pattern = max(horizontal_pattern, verticle_pattern, diagonal_1_pattern, diagonal_2_pattern)

	tmp_res = {'h': horizontal_status_str, 'hp': horizontal_pattern,
			   'v': verticle_status_str, 'vp': verticle_pattern,
			   'd1': diagonal_1_status_str, 'd1p': diagonal_1_pattern,
			   'd2': diagonal_2_status_str, 'd2p': diagonal_2_pattern,
			   'pattern': pattern}
	# print '['+str(i-PADDING_SIZE)+','+str(j-PADDING_SIZE)+']', horizontal_status_str, verticle_status_str, diagonal_1_status_str, diagonal_2_status_str, horizontal_pattern, verticle_pattern, diagonal_1_pattern, diagonal_2_pattern

	return pattern

def pos_legal_check(i, j):
	if i >= 0 and i < BOARD_SIZE and j >= 0 and j < BOARD_SIZE:
		return True
	else:
		return False

# update the given macro_feature_plane with new move (pos_id) in board_state
def update_macro_feature_plane(board_state, macro_feature_plane, player, pos_id):
	board_state = np.lib.pad(board_state, PADDING_SIZE, 'constant', constant_values=0)

	new_macro_feature_plane = copy.deepcopy(macro_feature_plane)

	# considered in the original BOARD_SIZE*BOARD_SIZE board
	i, j = next_move_id_pos_conversion(pos_id)
	# padded position
	ii, jj= i + PADDING_SIZE, j + PADDING_SIZE

	# print '('+str(i)+', '+str(j)+')'

	# horizontal
	for k in range(jj - PADDING_SIZE, jj + PADDING_SIZE + 1):
		if pos_legal_check(i, k - PADDING_SIZE):
			new_pattern = get_pos_pattern(board_state, ii, k, player)
			new_macro_feature_plane[i][k - PADDING_SIZE] = new_pattern
	# verticle
	for k in range(ii - PADDING_SIZE, ii + PADDING_SIZE + 1):
		if k != ii and pos_legal_check(k - PADDING_SIZE, j):
			new_pattern = get_pos_pattern(board_state, k, jj, player)
			new_macro_feature_plane[k - PADDING_SIZE][j] = new_pattern
	# diagonal_1
	for k in range(-PADDING_SIZE, PADDING_SIZE + 1):
		if k != 0 and pos_legal_check(ii + k - PADDING_SIZE, jj + k - PADDING_SIZE):
			new_pattern = get_pos_pattern(board_state, ii + k, jj + k, player)
			new_macro_feature_plane[(ii + k) - PADDING_SIZE][(jj + k) - PADDING_SIZE] = new_pattern
	# diagonal_2
	for k in range(-PADDING_SIZE, PADDING_SIZE + 1):
		if k != 0 and pos_legal_check(ii - k - PADDING_SIZE, jj + k - PADDING_SIZE):
			new_pattern = get_pos_pattern(board_state, ii - k, jj + k, player)
			new_macro_feature_plane[(ii - k) - PADDING_SIZE][(jj + k) - PADDING_SIZE] = new_pattern

	return new_macro_feature_plane

def get_object_micro_feature_plane(macro_feature_plane, object_feature):
	object_micro_feature_plane = np.zeros((BOARD_SIZE, BOARD_SIZE))
	object_micro_feature_plane[macro_feature_plane == object_feature] = 1
	return object_micro_feature_plane

# get the input to the NN, 3D imgae feature planes
def get_micro_feature_planes(macro_feature_plane):
	micro_feature_planes = []

	# connect-five
	connect_five = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['connect_five'])
	micro_feature_planes.append(connect_five)

	# alive-four
	alive_four = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['alive_four'])
	micro_feature_planes.append(alive_four)

	# sleep-four
	sleep_four = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['sleep_four'])
	micro_feature_planes.append(sleep_four)

	# alive-three
	alive_three = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['alive_three'])
	micro_feature_planes.append(alive_three)

	# sleep-three
	sleep_three = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['sleep_three'])
	micro_feature_planes.append(sleep_three)

	# alive-two
	alive_two = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['alive_two'])
	micro_feature_planes.append(alive_two)

	# sleep-two
	sleep_two = get_object_micro_feature_plane(macro_feature_plane, FEATURES_VALUE_MAP['sleep_two'])
	micro_feature_planes.append(sleep_two)

	micro_feature_planes = np.array(micro_feature_planes)
	return micro_feature_planes

def extract_feature_planes(data_list):	
	# for debugging
	# data_list.append(conversion('/Users/panling/Desktop/gomoku/gomoku_dataset/opening_1_380/0x1-25(1).psq'))

	# format of data_list: [[{'state': state, 'action': action, 'winner': winner}, ...], ... [{'state': state, 'action': action, 'winner': winner}]]
	macro_feature_plane_list, micro_feature_planes_list = [], []
	cnt = 0
	for game_record in data_list:
		print 'extracting feature planes for data ' + str(cnt)
		cnt += 1

		start = True
		move1_id, move2_id = -1, -1
		for record in game_record:
			state, action, winner = record['state'], record['action'], record['winner']

			if winner == 1:
				loser = 2
			else:
				loser = 1

			# get initial macro feature planes
			if start:
				macro_feature_plane = get_macro_feature_plane(state, winner)
				macro_feature_plane_loser = get_macro_feature_plane(state, loser)
				start = False
			# update macro feature planes: state_prev -> current player's move, opponent player's move -> state_curr
			else:
				macro_feature_plane_tmp = update_macro_feature_plane(state, macro_feature_plane_prev, winner, move1_id)
				macro_feature_plane_tmp_loser = update_macro_feature_plane(state, macro_feature_plane_prev_loser, loser, move1_id)
				if move2_id != -1:
					macro_feature_plane = update_macro_feature_plane(state, macro_feature_plane_tmp, winner, move2_id)
					macro_feature_plane_loser = update_macro_feature_plane(state, macro_feature_plane_tmp_loser, loser, move2_id)
				else:
					macro_feature_plane = copy.deepcopy(macro_feature_plane_tmp)
					macro_feature_plane_loser = copy.deepcopy(macro_feature_plane_tmp_loser)
			
			macro_feature_plane_prev = copy.deepcopy(macro_feature_plane)
			macro_feature_plane_prev_loser = copy.deepcopy(macro_feature_plane_loser)

			macro_feature_plane_combo = []
			macro_feature_plane_combo.append(macro_feature_plane_prev)
			macro_feature_plane_combo.append(macro_feature_plane_prev_loser)
			macro_feature_plane_combo = np.array(macro_feature_plane_combo)

			move1_id = action
			move2_id = record['transition_move']

			# raw macro feature plane
			# curr_macro_feature_plane = {'state': macro_feature_plane_combo, 'action': action}
			curr_macro_feature_plane = {'state': macro_feature_plane_prev, 'action': action}
			macro_feature_plane_list.append(curr_macro_feature_plane)

			# split macro feature plane into micro feature planes
			micro_feature_planes = get_micro_feature_planes(macro_feature_plane)
			micro_feature_planes_list.append({'state': micro_feature_planes, 'action': action})

	# return value
	return micro_feature_planes_list

# get (x_train, y_train) from training_dataset or (x_test, y_test, w_test) from testing dataset
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

# partition the dataset into (x_train, y_train, w_train), (x_test, y_test, w_test)
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

def shuffle_data(data_list):
	return random.shuffle(data_list)

def sl_training(data_list):
	# randomly shuffle the data
	shuffle_data(data_list)

	# get training and testing data
	x_train, y_train, x_test, y_test = partition_dataset(data_list)
	
	# rotate and mirror the training data
	# print 'rotating and mirroring...'
	# rotate_and_mirror(x_train, y_train)

	# preprocessing
	channels_cnt = len(FEATURES_VALUE_MAP)
	x_train = x_train.reshape(x_train.shape[0], BOARD_SIZE, BOARD_SIZE, channels_cnt)
	x_test = x_test.reshape(x_test.shape[0], BOARD_SIZE, BOARD_SIZE, channels_cnt)
	input_shape = (BOARD_SIZE, BOARD_SIZE, channels_cnt)

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

	num_of_intermediate_layers = 12
	model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
	for i in range(num_of_intermediate_layers):
		model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
	model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
	model.add(Flatten())
	model.add(keras.layers.core.Activation(activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
	
	# training
	batch_size = 128
	epochs = 12
	csv_logger = CSVLogger('training.log')
	model.fit(x_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test),
	          callbacks=[csv_logger])
	
	# testing
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

# TODO: rotation and reflection
if __name__ == '__main__':
	dataset_dir = os.path.abspath('gomoku_dataset')

	dataset_folder_dir_list = []
	for file in os.listdir(dataset_dir):
		file_dir = os.path.join(dataset_dir, file)
		if os.path.isdir(file_dir):
			dataset_folder_dir_list.append(file_dir)
	
	data_list = []
	for dataset_folder_dir in dataset_folder_dir_list:
		curr_data_list = read_file_in_folder(dataset_folder_dir)
		data_list.extend(curr_data_list)

	macro_feature_plane_list = extract_feature_planes(data_list)

	sl_training(macro_feature_plane_list)
	# extract_feature_planes()
