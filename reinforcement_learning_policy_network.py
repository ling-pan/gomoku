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

# opponent pool initialization
def init_opponent_pool():
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
	model.load_weights('sl_policy_network_weights.hdf5')
	print model.get_weights()
# # play the game between player1 and player2
# def self_play():
# 	# choose players from opponent pool
# 	# init opening board_state
# 	# play the game

# # update network weights
# def policy_gradient_update():
# 	# todo

# # opponent pool update
# def update_opponent_pool():
# 	# todo

# def run_one_game():
# 	# init opponent pool
# 	# self-play
# 	# replay to update network weights
# 	# update opponent pool

if __name__ == '__main__':
	# reinforment learning policy network
	init_opponent_pool()