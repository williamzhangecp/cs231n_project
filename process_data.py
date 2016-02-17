import csv
import numpy as np

def read_faces_csv(filename, center=True):
	"""
	Function that takes as input file a csv.reader() instance and assumes the following formatting:
	emotion, pixels (2034 of them), usage (train, test, val)
	Returns the following numpy arrays:
	- X_train, y_train (respectively (N, 48, 48), (N,) representing raw grayscale pixels and emotion labels)
	- X_test, y_test
	- X_val, y_val
	"""

        csv_file = open(filename)

        reader_file = csv.reader(csv_file)

	# Discard header
	row = next(reader_file)

	X_train_list, y_train_list = [], []
	X_test_list, y_test_list = [], []
	X_val_list, y_val_list = [], []

	N_train, N_test, N_val = 0, 0, 0

	for row in reader_file:
		y_str, X_row_str, data_type = row
		y = int(y_str)

		X_row_strs = X_row_str.split(' ')
		X_row = [float(x) for x in X_row_strs]

		if data_type == 'PublicTest':
			y_test_list.append(y)
			X_test_list.append(X_row)
			N_test += 1
		elif data_type == 'PrivateTest':
			y_val_list.append(y)
			X_val_list.append(X_row)
			N_val += 1
		else:
			y_train_list.append(y)
			X_train_list.append(X_row)
			N_train += 1

	X_train = np.asarray(X_train_list).astype('float64').reshape((N_train, 48, 48))
	y_train = np.asarray(y_train_list)

	X_test = np.asarray(X_test_list).astype('float64').reshape((N_test, 48, 48))
	y_test = np.asarray(y_test_list)

	X_val = np.asarray(X_val_list).astype('float64').reshape((N_val, 48, 48))
	y_val = np.asarray(y_val_list)

	# decide to mean-center or not
	if center:
		train_mean = X_train.mean(axis = 0)
		X_train -= train_mean
		X_test -= train_mean
		X_val -= train_mean

	#########
	return X_train, y_train, X_test, y_test, X_val, y_val
