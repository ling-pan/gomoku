# -*- coding:utf-8 -*-

import numpy as np
import copy, os, random, argparse, json

import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D
from keras.callbacks import CSVLogger

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
		if i < 10:
			print '[0' + str(i) + '] ', 
		else:
			print '['+str(i)+'] ', 
		for j in range(col_cnt):
			print int(matrix[i][j]), '  ',
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

def read_file_in_folder(folder_dir, with_feature_planes):
	if with_feature_planes:
		cnt = 0
		state_action_pair_list = []
		for file in os.listdir(folder_dir):
			file_dir = os.path.join(folder_dir, file)
			if os.path.splitext(file_dir)[1] == '.psq':
				curr_state_action_pair_list = conversion(file_dir)
				state_action_pair_list.append(curr_state_action_pair_list)

				# # debug
				# if cnt > 0:
				# 	break
				# cnt += 1

		return state_action_pair_list
	else:
		cnt = 0
		state_action_pair_list = []
		for file in os.listdir(folder_dir):
			file_dir = os.path.join(folder_dir, file)
			if os.path.splitext(file_dir)[1] == '.psq':
				curr_state_action_pair_list = conversion(file_dir)
				state_action_pair_list.extend(curr_state_action_pair_list)

				# # debug
				# if cnt > 0:
				# 	break
				# cnt += 1

		return state_action_pair_list		

# connect-five:7 alive-four:6 sleep-four:5 alive-three:4 sleep-three:3 alive-two:2 sleep-two:1 none:0
def check_match_pattern(status):
	c5, a4, s4, a3, s3, a2, s2 = 0, 0, 0, 0, 0, 0, 0

	# connect-five
	for pattern in FEATURES_MAP['connect_five']:
		if pattern in status:
			c5 = 1
			break
	# alive-four
	for pattern in FEATURES_MAP['alive_four']:
		if pattern in status:
			a4 = 1
			break
	# sleep-four
	for pattern in FEATURES_MAP['sleep_four']:
		if pattern in status:
			s4 = 1
			break
	# alive-three
	for pattern in FEATURES_MAP['alive_three']:
		if pattern in status:
			a3 = 1
			break
	# sleep-three
	for pattern in FEATURES_MAP['sleep_three']:
		if pattern in status:
			s3 = 1
			break
	# alive-two
	for pattern in FEATURES_MAP['alive_two']:
		if pattern in status:
			a2 = 1
	# sleep-two
	for pattern in FEATURES_MAP['sleep_two']:
		if pattern in status:
			s2 = 1

	res = (c5, a4, s4, a3, s3, a2, s2)
	return res

def print_sub_board_state(board_state):
	for i in range(BOARD_SIZE):
		for j in range(BOARD_SIZE):
			print int(board_state[i + PADDING_SIZE][j + PADDING_SIZE]), '\n'
		print '\n'
	print '\n'

def get_macro_feature_plane(board_state, player):
	board_state = np.lib.pad(board_state, PADDING_SIZE, 'constant', constant_values=-1)

	macro_feature_plane = [[0 for i in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]

	st, ed = PADDING_SIZE, BOARD_SIZE + PADDING_SIZE
	for i in range(st, ed):
		for j in range(st, ed):
			curr_pattern = {'connect_five': 0, 'alive_four': 0, 'sleep_four': 0, 'alive_three': 0, 'sleep_three': 0, 'alive_two': 0, 'sleep_two': 0}
			if board_state[i][j] == 0:
				# assume that we place player's move at position (i, j)
				board_state[i][j] = player
				# print 'assumption move at (', i - PADDING_SIZE, j - PADDING_SIZE, '):'
				# print_sub_board_state(board_state)
				# format of curr_patter: {'c5': c5_cnt, 'a4': a4_cnt, 's4': s4_cnt, 'a3': a3_cnt, 's3': s3_cnt, 'a2': a2_cnt, 's2': s2_cnt'}
				curr_pattern = get_pos_pattern(board_state, i, j, player)
				# print curr_pattern
				# revert to original
				board_state[i][j] = 0
				# print 'reverting to original:'
				# print_sub_board_state(board_state)
			macro_feature_plane[i - PADDING_SIZE][j - PADDING_SIZE] = curr_pattern

	return macro_feature_plane

def get_summarized_pattern(horizontal_pattern, verticle_pattern, diagonal_1_pattern, diagonal_2_pattern):
	num_of_features = len(FEATURES_MAP)
	cnt_list = []
	for i in range(num_of_features):
		curr_cnt = horizontal_pattern[i] + verticle_pattern[i] + diagonal_1_pattern[i] + diagonal_2_pattern[i]
		cnt_list.append(curr_cnt)
	return cnt_list

# !!! assuming that the board_state is properly padded !!!
def get_pos_pattern(board_state, i, j, player):
	# get opponent_player
	if player == 1:
		opponent_player = 2
	else:
		opponent_player = 1

	# horizontal
	horizontal_status_str = ''
	for k in range(j - PADDING_SIZE, j + (PADDING_SIZE + 1)):
		if board_state[i][k] == player:
			horizontal_status_str += 'x'
		elif board_state[i][k] == EMPTY:
			horizontal_status_str += '_'
		elif board_state[i][k] == opponent_player:
			horizontal_status_str += 'o'
	horizontal_pattern = check_match_pattern(horizontal_status_str)

	# verticle
	verticle_status_str = ''
	for k in range(i - PADDING_SIZE, i + (PADDING_SIZE + 1)):
		if board_state[k][j] == player:
			verticle_status_str += 'x'
		elif board_state[k][j] == EMPTY:
			verticle_status_str += '_'
		elif board_state[k][j] == opponent_player:
			verticle_status_str += 'o'
	verticle_pattern = check_match_pattern(verticle_status_str)

	# diagonal_1
	diagonal_1_status_str = ''
	for k in range(PADDING_SIZE, 0, -1):
		if board_state[i - k][j - k] == player:
			diagonal_1_status_str += 'x'
		elif board_state[i - k][j - k] == EMPTY:
			diagonal_1_status_str += '_'
		elif board_state[i - k][j - k] == opponent_player:
			diagonal_1_status_str += 'o'
	for k in range(PADDING_SIZE + 1):
		if board_state[i + k][j + k] == player:
			diagonal_1_status_str += 'x'
		elif board_state[i + k][j + k] == EMPTY:
			diagonal_1_status_str += '_'
		elif board_state[i + k][j + k]== opponent_player:
			diagonal_1_status_str += 'o'
	diagonal_1_pattern = check_match_pattern(diagonal_1_status_str)

	# diagonal_2
	diagonal_2_status_str = ''
	for k in range(PADDING_SIZE, 0, -1):
		if board_state[i + k][j - k] == player:
			diagonal_2_status_str += 'x'
		elif board_state[i + k][j - k] == EMPTY:
			diagonal_2_status_str += '_'
		elif board_state[i + k][j - k] == opponent_player:
			diagonal_2_status_str += 'o'
	for k in range(PADDING_SIZE + 1):
		if board_state[i - k][j + k] == player:
			diagonal_2_status_str += 'x'
		elif board_state[i - k][j + k] == EMPTY:
			diagonal_2_status_str += '_'
		elif board_state[i - k][j + k] == opponent_player:
			diagonal_2_status_str += 'o'
	diagonal_2_pattern = check_match_pattern(diagonal_2_status_str)

	summarized_pattern = get_summarized_pattern(horizontal_pattern, verticle_pattern, diagonal_1_pattern, diagonal_2_pattern)
	c5_cnt = summarized_pattern[0]
	a4_cnt = summarized_pattern[1]
	s4_cnt = summarized_pattern[2]
	a3_cnt = summarized_pattern[3]
	s3_cnt = summarized_pattern[4]
	a2_cnt = summarized_pattern[5]
	s2_cnt = summarized_pattern[6]
	pattern = {'connect_five': c5_cnt, 'alive_four': a4_cnt, 'sleep_four': s4_cnt, 'alive_three': a3_cnt, 'sleep_three': s3_cnt, 'alive_two': a2_cnt, 'sleep_two': s2_cnt}

	return pattern

def pos_legal_check(i, j):
	if i >= 0 and i < BOARD_SIZE and j >= 0 and j < BOARD_SIZE:
		return True
	else:
		return False

# update the given macro_feature_plane with new move (pos_id) in board_state
def update_macro_feature_plane(board_state, original_macro_feature_plane, player, pos_id):
	board_state = np.lib.pad(board_state, PADDING_SIZE, 'constant', constant_values=-1)

	new_macro_feature_plane = copy.deepcopy(original_macro_feature_plane)

	# id to position conversion
	ii, jj = next_move_id_pos_conversion(pos_id)
	i, j = ii + PADDING_SIZE, jj + PADDING_SIZE

	# update self
	new_macro_feature_plane[ii][jj] = {'connect_five': 0, 'alive_four': 0, 'sleep_four': 0, 'alive_three': 0, 'sleep_three': 0, 'alive_two': 0, 'sleep_two': 0}
	# horizontal
	for k in range(j - PADDING_SIZE, j + PADDING_SIZE + 1):
		if board_state[i][k] == 0 and pos_legal_check(i - PADDING_SIZE, k - PADDING_SIZE):
			# assume that we place player's move at position (i, k)
			board_state[i][k] = player
			# update the feature of position (i, k)
			curr_pattern = get_pos_pattern(board_state, i, k, player)
			new_macro_feature_plane[i - PADDING_SIZE][k - PADDING_SIZE] = curr_pattern
			# revert to original
			board_state[i][k] = 0
	# verticle
	for k in range(i - PADDING_SIZE, i + PADDING_SIZE + 1):
		if board_state[k][j] == 0 and pos_legal_check(k - PADDING_SIZE, j - PADDING_SIZE):
			# assume that we place player's move at position (i, k)
			board_state[k][j] = player
			# update the feature of position (k, j)
			curr_pattern = get_pos_pattern(board_state, k, j, player)
			new_macro_feature_plane[k - PADDING_SIZE][j - PADDING_SIZE] = curr_pattern
			# revert to original
			board_state[k][j] = 0
	# diagonal_1
	for k in range(-PADDING_SIZE, PADDING_SIZE + 1):
		if board_state[i + k][j + k] == 0 and pos_legal_check(i + k - PADDING_SIZE, j + k - PADDING_SIZE):
			# assume that we place player's move at position (i + k, j + k)
			board_state[i + k][j + k] = player
			# update the feature of position (i + k, j + k)
			curr_pattern = get_pos_pattern(board_state, i + k, j + k ,player)
			new_macro_feature_plane[i + k - PADDING_SIZE][j + k - PADDING_SIZE] = curr_pattern
			# revert to original
			board_state[i + k][j + k] = 0
	# diagonal_2
	for k in range(-PADDING_SIZE, PADDING_SIZE + 1):
		if board_state[i - k][j + k] == 0 and pos_legal_check(i - k - PADDING_SIZE, j + k - PADDING_SIZE):
			# assume that we place player's move at position (i - k, j + k)
			board_state[i - k][j + k] = player
			# update the feature of position (i - k, j + k)
			curr_pattern = get_pos_pattern(board_state, i - k, j + k, player)
			new_macro_feature_plane[i - k - PADDING_SIZE][j + k - PADDING_SIZE] = curr_pattern
			# revert to original
			board_state[i - k][j + k] = 0

	return new_macro_feature_plane

# get the input to the NN, 3D imgae feature planes
def get_micro_feature_planes(macro_feature_plane):
	micro_feature_planes = np.zeros((len(FEATURES_MAP), BOARD_SIZE, BOARD_SIZE))
	for i in range(BOARD_SIZE):
		for j in range(BOARD_SIZE):
			curr_feature_map = macro_feature_plane[i][j]
			# connect-five
			micro_feature_planes[0][i][j] = curr_feature_map['connect_five']
			# alive-four
			micro_feature_planes[1][i][j] = curr_feature_map['alive_four']
			# sleep-four
			micro_feature_planes[2][i][j] = curr_feature_map['sleep_four']
			# alive-three
			micro_feature_planes[3][i][j] = curr_feature_map['alive_three']
			# sleep-three
			micro_feature_planes[4][i][j] = curr_feature_map['sleep_three']
			# alive-two
			micro_feature_planes[5][i][j] = curr_feature_map['alive_two']
			# sleep-two
			micro_feature_planes[6][i][j] = curr_feature_map['sleep_two']
			# debug
			# print '(', i, j, '): ', curr_feature_map
			# print int(micro_feature_planes[0][i][j]), int(micro_feature_planes[1][i][j]), int(micro_feature_planes[2][i][j]), int(micro_feature_planes[3][i][j]), int(micro_feature_planes[4][i][j]), int(micro_feature_planes[5][i][j]), int(micro_feature_planes[6][i][j])
	return micro_feature_planes

def print_board_state(matrix):
	row_cnt, col_cnt = matrix.shape
	for i in range(row_cnt):
		if i < 10:
			print '[0' + str(i) + '] ', 
		else:
			print '[' + str(i) + '] ', 
		for j in range(col_cnt):
			if matrix[i][j] == 0:
				print '-',
			elif matrix[i][j] == 1:
				print '*',
			elif matrix[i][j] == 2:
				print 'o',
			print '  ',
		print '\n'
	print '\n'

def extract_feature_planes(data_list):	
	# # for debugging
	# data_list = []
	# data_list.append(conversion('/Users/panling/Desktop/gomoku/gomoku_dataset/opening_1_380/0x1-25(1).psq'))

	micro_feature_planes_list = []
	data_to_save_list = []
	cnt = 0
	for game_record in data_list:
		print 'extracting feature planes for data ' + str(cnt)
		cnt += 1

		start = True
		move1_id, move2_id = -1, -1

		# for debugging
		tmp_cnt = 0
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
				if move1_id == -1:
					break
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

			# split macro feature plane into micro feature planes
			micro_feature_planes = get_micro_feature_planes(macro_feature_plane_prev)
			micro_feature_planes_loser = get_micro_feature_planes(macro_feature_plane_prev_loser)

			# write feature planes data
			micro_feature_planes_l = micro_feature_planes.tolist()
			micro_feature_planes_loser_l = micro_feature_planes_loser.tolist()
			state_l = state.tolist()
			feature_planes_list = []
			feature_planes_list.extend(micro_feature_planes_l)
			feature_planes_list.extend(micro_feature_planes_loser_l)
			feature_planes_list.append(state_l)
			curr_data_to_save = {'state': feature_planes_list, 'action': action}
			data_to_save_list.append(curr_data_to_save)

			micro_feature_planes_combo = []
			micro_feature_planes_combo.extend(micro_feature_planes)
			micro_feature_planes_combo.extend(micro_feature_planes_loser)
			micro_feature_planes_combo.append(state)
			
			micro_feature_planes_list.append({'state': micro_feature_planes_combo, 'action': action})

			# # debug
			# print 'board_state:'
			# print_board_state(state)

			# print 'feature planes for player 1:'
			# idx = 0
			# feature_name_list = ['connect_five', 'alive_four', 'sleep_four', 'alive_three', 'sleep_three', 'alive_two', 'sleep_two']
			# for i in range(len(FEATURES_MAP)):
			# 	print feature_name_list[i]
			# 	print_np_matirx(micro_feature_planes[idx])
			# 	idx += 1
			# 	print '\n'

			# print 'feature planes for player 2:'
			# idx = 0
			# feature_name_list = ['connect_five', 'alive_four', 'sleep_four', 'alive_three', 'sleep_three', 'alive_two', 'sleep_two']
			# for i in range(len(FEATURES_MAP)):
			# 	print feature_name_list[i]
			# 	print_np_matirx(micro_feature_planes_loser[idx])
			# 	idx += 1
			# 	print '\n'
			# tmp_cnt += 1

	# save data
	with open('feature_planes_data.json', 'w') as json_file:
		json_file.write(json.dumps(data_to_save_list))

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

def sl_training(data_list, channels_cnt):
	# randomly shuffle the data
	shuffle_data(data_list)

	# get training and testing data
	x_train, y_train, x_test, y_test = partition_dataset(data_list)
	
	# rotate and mirror the training data
	# print 'rotating and mirroring...'
	# rotate_and_mirror(x_train, y_train)

	print len(data_list)
	print x_train.shape

	# preprocessing
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
	# save weights
	model.save_weights('sl_policy_network_weights.hdf5')

	# testing
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

# TODO: rotation and reflection
if __name__ == '__main__':
	# argument assignment
	parser = argparse.ArgumentParser()
	parser.add_argument('--with_feature_planes', dest = 'with_feature_planes', help = 'train the network with feature planes or not')
	parser.add_argument('--feature_planes_file', dest = 'feature_planes_file', help = 'feature planes file processed in advance')
	args = parser.parse_args()
	with_feature_planes = int(args.with_feature_planes) 
	feature_planes_file = args.feature_planes_file

	# data preprocessing
	if feature_planes_file == 'none':
		dataset_dir = os.path.abspath('gomoku_dataset')

		dataset_folder_dir_list = []
		for file in os.listdir(dataset_dir):
			file_dir = os.path.join(dataset_dir, file)
			if os.path.isdir(file_dir):
				dataset_folder_dir_list.append(file_dir)
		
		data_list = []
		for dataset_folder_dir in dataset_folder_dir_list:
			curr_data_list = read_file_in_folder(dataset_folder_dir, with_feature_planes)
			data_list.extend(curr_data_list)
	else:
		formatted_data_list = []
		with open(feature_planes_file, 'r') as json_file:
			data_list = json.load(json_file)
			for data in data_list:
				state = np.array(data['state'])
				action = data['action']
				formatted_data = {'state': state, 'action': action}
				formatted_data_list.append(formatted_data)
		data_list = formatted_data_list

	# train the supervised learning policy network
	if with_feature_planes == 1:
		channels_cnt = len(FEATURES_MAP) * 2 + 1
		feature_planes = extract_feature_planes(data_list)
		sl_training(feature_planes, channels_cnt)
	else:
		channels_cnt = 1
		sl_training(data_list, channels_cnt)