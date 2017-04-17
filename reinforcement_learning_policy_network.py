# -*- coding:utf-8 -*-

import numpy as np
import copy, os
import random
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Flatten, Conv2D
from keras.callbacks import CSVLogger
from rotation_and_reflection import *
import keras.backend as K

BOARD_SIZE = 20
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE
opening_state_list = []

def log_loss(y_true, y_pred):
    return -y_true * K.log(K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon()))

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
	# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer)
	model.compile(loss=log_loss, optimizer=optimizer)
	
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
			row, col = int(move[0]), int(move[1])
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

def move_leagl_check(i, j):
	return (i > 0 and i < BOARD_SIZE and j > 0 and j < BOARD_SIZE)

# win: 1, tie:0, nothing happens: -1
def judge_winning_state(board_state, move_id):
	move_i, move_j = move_id_pos_conversion(move_id)

	if board_state[move_i][move_j] == 1:
		player = 1
		opponent_player = 2
	elif board_state[move_i][move_j] == 2:
		player = 2
		opponent_player = 1

	i, j = move_i, move_j

	# horizontal
	horizontal_pattern_str = ''
	for k in range(-4, 4):
		if move_leagl_check(i, j - k):
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
		if move_leagl_check(i - k, j):
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
		if move_leagl_check(i - k, j + k):
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
		if move_leagl_check(i + k, j + k):
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

def print_np_matrix(matrix):
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

def choose_move_from_prob_distribution(prob_distri, board_state):
	to_continue_choosing = True
	while to_continue_choosing:
		current_choice = np.where(prob_distri == max(prob_distri))[0][0]
		i, j = move_id_pos_conversion(current_choice)
		if board_state[i][j] != 0:
			prob_distri[current_choice] = -1
		else:
			to_continue_choosing = False
			final_choice = current_choice
	return final_choice

def run_one_batch_game(current_player_policy_network, opponent_pool, mini_batch_size, optimizer):
	# player 1: current player, player 2: opponent player
	opponent_player_policy_network = np.random.choice(opponent_pool)

	win_cnt = 0

	states, moves = [[] for _ in range(mini_batch_size)], [[] for _ in range(mini_batch_size)]

	board_states = np.zeros((mini_batch_size, BOARD_SIZE, BOARD_SIZE))
	opening_state_idx_list = [random.randint(0, len(opening_state_list) - 1) for _ in range(mini_batch_size)]
	current_player_color_list = []
	for i in range(mini_batch_size):
		opening_moves_list = opening_state_list[opening_state_idx_list[i]]
		color = 1 # player 1 plays the first move
		for opening_move in opening_moves_list:
			# get move
			opening_move_row, opening_move_col = opening_move
			# do move
			board_states[i][opening_move_row][opening_move_col] = color
			color = change_color(color)
		current_player_color_list.append(color)

	# self-play: record the state-action pairs and final reward
	unfinished_games_id_list = range(mini_batch_size)
	reward_list = [0 for i in range(mini_batch_size)]
	while (len(unfinished_games_id_list) > 0):
		for game_id in unfinished_games_id_list:
			# get current basic stats
			to_learn = False
			current_color = current_player_color_list[game_id]
			if current_color == 1:
				current_player = current_player_policy_network
				to_learn = True
			else:
				current_player = opponent_player_policy_network

			# predict next move
			format_board_state = [board_states[game_id]]
			format_board_state = np.array(format_board_state)
			format_board_state = format_board_state.reshape(format_board_state.shape[0], BOARD_SIZE, BOARD_SIZE, 1)

			predicted_move_prob_distribution = current_player.predict(format_board_state)[0]
			predicted_move_id = choose_move_from_prob_distribution(predicted_move_prob_distribution, board_states[game_id])

			# save state-action pair for the learner
			if to_learn:
				states[game_id].append(board_states[game_id])
				moves[game_id].append(predicted_move_id)

			# do move
			move_row, move_col = move_id_pos_conversion(predicted_move_id)
			board_states[game_id][move_row][move_col] = current_color
			current_player_color_list[game_id] = change_color(current_color)

			curr_res = judge_winning_state(board_states[game_id], predicted_move_id)
			if curr_res != -1:
				unfinished_games_id_list.remove(game_id)
				if curr_res == 1:
					# curr_player wins
					if current_player_color_list[game_id] == 2:
						win_cnt += 1
						reward_list[game_id] = 1
					else:
						reward_list[game_id] = -1
				else:
					# tie
					reward_list[game_id] = 0

	# "replay" to update network weights
	for i in range(mini_batch_size):
		reward = reward_list[i]
		curr_states = states[i]
		curr_moves = moves[i]

		curr_states = np.array(curr_states)
		curr_states = curr_states.reshape(curr_states.shape[0], BOARD_SIZE, BOARD_SIZE, 1)
		curr_moves = np.array(curr_moves)
		curr_moves = keras.utils.to_categorical(curr_moves, NUM_CLASSES)

		optimizer.lr = K.abs(optimizer.lr) * reward
		current_player_policy_network.train_on_batch(curr_states, curr_moves)
	
	win_ratio = (1.0 * win_cnt) / mini_batch_size * 100
	print 'win ratio =', win_ratio, '%'

	return current_player_policy_network

# evaluate rl policy network by playing EVAL_SIZE games with initial sl_policy network
def eval(opponent_pool, eval_size):
	print 'Evaluating:'
	
	# player 1: rl policy network; player 2: sl policy network
	sl_policy_network = opponent_pool[0]
	rl_policy_network = opponent_pool[len(opponent_pool) - 1]

	win_ratio = 0

	for i in range(eval_size):
		print 'game[ '+ str(i) + ']: ',

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
			current_player = rl_policy_network
		elif color == 2:
			current_player = sl_policy_network

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

			predicted = current_player.predict(format_board_state)[0]
			predicted_move_id = choose_move_from_prob_distribution(predicted, board_state)

			move_row, move_col = move_id_pos_conversion(predicted_move_id)

			# perform current move
			board_state[move_row][move_col] = color
			color = change_color(color)

			# swap players
			if color == 1:
				current_player = rl_policy_network
			elif color == 2:
				current_player = sl_policy_network

			# check whether to continue
			res = judge_winning_state(board_state, predicted_move_id)
			if res != -1:
				continue_the_game = False
				if res == 0:
					winner = 0
				else:
					if color == 1:
						winner = 2
					else:
						winner = 1

		# calculate win ration
		if winner == 1:
			print 'won!!!'
			win_ratio += 1
		else:
			print 'lost...'

	win_ratio = (1.0 * win_ratio / eval_size) * 100
	print '---------- Win Ratio: ', win_ratio, '----------'

def reinforment_learning(num_of_iterations):
	# init opponent pool
	optimizer = optimizers.SGD()
	opponent_pool = init_opponent_pool('sl_policy_network_weights.hdf5', optimizer)

	current_player = opponent_pool[0]

	# play one batch game
	mini_batch_size = 128
	for i in range(num_of_iterations):
		print 'iteration ' + str(i) + '...'
		current_player = run_one_batch_game(current_player, opponent_pool, mini_batch_size, optimizer)
		if i != 0 and i % 50 == 0:
			np.append(opponent_pool, current_player)

	eval_size = 100
	eval(opponent_pool, eval_size)

if __name__ == '__main__':
	parent_dir = os.path.abspath('')
	opening_state_list = load_opening_file(os.path.join(parent_dir, 'openings.txt'))

	num_of_iterations = 1000
	reinforment_learning(num_of_iterations)

