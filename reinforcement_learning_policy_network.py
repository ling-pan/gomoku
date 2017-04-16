# -*- coding:utf-8 -*-

import numpy as np
import copy, os
import random
import keras
from keras.models import Sequential
from keras.layers import Flatten, Conv2D
from keras.callbacks import CSVLogger
from rotation_and_reflection import *

BOARD_SIZE = 20
opening_file_list = []

# opponent pool initialization
def init_opponent_pool(init_weights_dir, optimizer):
	model = Sequential()
	num_of_intermediate_layers = 12
	channels_cnt = 1
	input_shape = (BOARD_SIZE, BOARD_SIZE, channels_cnt)
	model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
	for i in range(num_of_intermediate_layers):
		model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
	model.add(Conv2D(1, kernel_size=(1, 1), padding='same'))
	model.add(Flatten())
	model.add(keras.layers.core.Activation(activation='softmax'))

	model.load_weights(init_weights_dir)
    model.compile(loss='keras.losses.categorical_crossentropy', optimizer=optimizer)

	opponent_pool = []
	opponent_pool.append(model)
	opponent_pool = np.array(opponent_pool)

	return opponent_pool

def load_opening_file(opening_file_dir):
	f = open(opening_file_dir, 'r')  
	content = f.readlines()  
	move_list = []
	for moves in content:
		moves = moves.split(' ')
		curr_move_list = []
		for move in moves:
			move = move.split(',')
			row, col = move[0], move[1]
			converted_move = (9 - row, 10 + col)
			curr_move_list.append(converted_move)
		move_list.append(curr_move_list)
	# move_list data format: [(r1, c1), ..., (rn, cn)]
	return move_list

def change_color(color):
	if color == 1:
		color = 2
	else:
		color = 1
	return color

def move_id_pos_conversion(move_id):
	row = move_id / BOARD_SIZE
	col = move_id % BOARD_SIZE
	return row, col

def move_pos_id_conversion(i, j):
	move_id = i * BOARD_SIZE + j
	return move_id

def check_winning_pattern(pattern_str):
	winning_pattern_str = 'xxxxx'
	return winning_pattern_str in pattern_str

# win: 1, tie:0, nothing happens: -1
def judge_winning_state(board_state, move_id):
	move_i, move_j = move_id_pos_conversion(move_id)

	if board_state[move_i][move_j] == 1:
		player = 1
		opponent_player = 2
	elif board_state[move_i][move_j] == 2:
		player = 2
		opponent_player = 1

	# horizontal
	horizontal_pattern_str = ''
	for k in range(-4, 4):
		if board_state[i][j - k] == 0:
			horizontal_pattern_str += '-'
		elif board_state[i][j - k] == player:
			horizontal_pattern_str += 'x'
		elif board_state[i][j - k] == opponent_player:
			horizontal_pattern_str += 'o'
	if check_winning_pattern(horizontal_pattern_str):
		return 1

	# verticle
	verticle_pattern_str = ''
	for k in range(-4, 4):
		if board_state[i - k][j] == 0:
			verticle_pattern_str += '-'
		elif board_state[i - k][j] == player:
			verticle_pattern_str += 'x'
		elif board_state[i - k][j] == opponent_player:
			verticle_pattern_str += 'o'
	if check_winning_pattern(verticle_pattern_str):
		return 1

	# diagonal_1
	diagonal_1_pattern_str = ''
	for k in range(-4, 4):
		if board_state[i - k][j + k] == 0:
			diagonal_1_pattern_str += '-'
		elif board_state[i - k][j + k] == player:
			diagonal_1_pattern_str += 'x'
		elif board_state[i - k][j + k] == opponent_player:
			diagonal_1_pattern_str += 'o'
	if check_winning_pattern(diagonal_1_pattern_str):
		return 1

	# diagonal_2
	diagonal_2_pattern_str = ''
	for k in range(-4, 4):
		if board_state[i + k][j + k] == 0:
			diagonal_2_pattern_str += '-'
		elif board_state[i + k][j + k] == player:
			diagonal_2_pattern_str += 'x'
		elif board_state[i + k][j + k] == opponent_player:
			diagonal_2_pattern_str += 'o'
	if check_winning_pattern(diagonal_2_pattern_str):
		return 1

	if np.sum(board_state == 0) == 0:
		return 0

	return -1 

# play the game between current policy network(player 1) and a randomly choosen previous policy network(player 2)
def run_one_batch_game(optimizer, current_policy_network, opponent_pool, mini_batch_size):
	# choose players from opponent pool
	prev_policy_network = np.random.choice(opponent_pool)

	win_ratio = 0

	for i in range(mini_batch_size):
		# init board_state
		board_state = np.zeros((BOARD_SIZE, BOARD_SIZE))
		color = 1

		# open with randomly chosen opening board_state
		opening_state = random.choice(opening_state_list)
		for opening_move in opening_state:
			opening_row, opening_col = opening_move
			board_state[opening_row][opening_col] = color
			color = change_color(color)

		if color == 1:
			current_player = current_policy_network
		elif color == 2:
			current_player = prev_policy_network

		# self-play the game to obtain reward
		continue_the_game = True
		winner = -1
		state_list, move_list = [], []
		while continue_the_game:
			# get move using current player's policy network
			format_board_state = []
			format_board_state.append(board_state)
			format_board_state = np.array(format_board_state)
			format_board_state = format_board_state.reshape(format_board_state.shape[0], BOARD_SIZE, BOARD_SIZE, 1)

			predicted = model.predict(format_board_state)
			predicted_move_id = np.where(prediced_val == max(prediced_val))[0][0]

			move_row, move_col = move_id_pos_conversion(move_id)

			# save current data
			state_list.append(board_state)
			move_list.append(move_id)

			# perform current move
			board_state[move_row][move_col] = color
			color = change_color(color)

			# check whether to continue
			res = judge_winning_state(board_state, move_id)
			if res != -1:
				continue_the_game = False
				if res == 0:
					winner = 0
				else:
					if color == 1:
						winner = 2
					else:
						winner = 1

		reward = 0
		if winner == 1:
			reward = 1
		else:
			reward = -1

		# policy gradient method to update network weights
		optimizer.lr = abs(optimizer.lr) * reward

		states = np.array(state_list)
		states = states.reshape(states.shape[0], BOARD_SIZE, BOARD_SIZE, 1)
		moves = np.array(move_list)
		model.train_on_batch(states, moves)

		# calculate win ration
		if winner == 1:
			win_ratio += 1

	win_ratio = win_ratio / mini_batch_size * 100
	print 'win_ratio: ', win_ratio

# opponent pool update
def update_opponent_pool(opponent_pool, new_rl_policy_network):
	opponent_pool.append(new_rl_policy_network)
	return opponent_pool

def reinforment_learning(num_of_iterations):
	parent_dir = os.path.abspath('gomoku')
	opening_file_list = load_opening_file(os.path.join(parent_dir, 'openings.txt'))

	# init opponent pool
	optimizer = keras.optimizers.SGD
	opponent_pool = init_opponent_pool('sl_training_weights.hdf5', optimizer)

	current_player = opponent_pool[0]

	# play one batch game
	mini_batch_size = 128
	for i in range(num_of_iterations):
		run_one_batch_game(optimizer, current_player, opponent_pool, mini_batch_size)

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

def get_winner(file_dir):
	step_cnt = int(file_dir.split('-')[1].split('(')[0])

	if step_cnt % 2 == 1:
		winner = 1
	else:
		winner = 2

	return winner

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

if __name__ == '__main__':
	reinforment_learning()