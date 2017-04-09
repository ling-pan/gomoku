def Rotate_90(board, next_move):
	board_rotate_90 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_90[j][n - 1 - i] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_90 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_90, next_move_rotate_90

def Rotate_180(board, next_move):
	board_rotate_180 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_180[n - 1 - i][n - 1 - j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_180 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_180, next_move_rotate_180

def Rotate_270(board, next_move):
	board_rotate_270 = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_rotate_270[n - 1 - j][i] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_rotate_270 = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_rotate_270, next_move_rotate_270

def Mirror_y_axis(board, next_move):
	board_mirror_y = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_mirror_y[i][n - 1 - j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_mirror_y = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_mirror_y, next_move_mirror_y

def Mirror_x_axis(board, next_move):
	board_mirror_x = np.zeros((BOARD_SIZE, BOARD_SIZE))
	
	n = BOARD_SIZE

	for i in range(n):
		for j in range(n):
			board_mirror_x[n - 1 - i][j] = board[i][j]

	next_move_i, next_move_j = next_move_id_pos_conversion(next_move)
	next_move_mirror_x = next_move_pos_id_conversion(next_move_i, next_move_j)

	return board_mirror_x, next_move_mirror_x

# rotate the board in 4 directions and mirror the board in 2 directions to get additional 8 transformations
def Rotate_and_mirror(x_train, y_train):
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