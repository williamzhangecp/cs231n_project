import csv
import h5py
import numpy as np
import sys

def read_faces_csv(filename, num_lines = None, mirror = False, center=True, \
					use_tf=False, use_th=False):
	"""
	Function that takes as input a filname to a csv file that assumes the following formatting:
	emotion, pixels (2034 of them), usage (train, test, val)
	Input variables:
	- num_lines: number of lines to read
	- mirror: indicates whether or not to horizontally flip the images
	- center: indicates whether or not to subtract the mean image from each image

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

	if num_lines == None:
		max_lines = sys.maxint
	else:
		max_lines = num_lines
	lines_read = 0
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
		lines_read += 1
		if lines_read >= max_lines:
			break

	X_train = np.asarray(X_train_list).astype('float64').reshape((N_train, 48, 48))
	y_train = np.asarray(y_train_list)

	X_test = np.asarray(X_test_list).astype('float64').reshape((N_test, 48, 48))
	y_test = np.asarray(y_test_list)

	X_val = np.asarray(X_val_list).astype('float64').reshape((N_val, 48, 48))
	y_val = np.asarray(y_val_list)

	# decide to flip or not
	if mirror:
		X_train = np.concatenate([X_train, X_train[:,:,::-1]])

	# decide to mean-center or not
	if center:
		train_mean = X_train.mean()
		train_std = X_train.std()
		X_train -= train_mean
		X_train /= train_std
		X_test -= train_mean
		X_test /= train_std
		X_val -= train_mean
		X_val /= train_std

	if use_th: # LUA (therefore Torch) uses 1-based indexing!!!
		y_train += 1
		y_test += 1
		y_val += 1

	############
        if use_tf: # Shape of array is different in tensorFlow
            return X_train[:,:,:,np.newaxis], y_train, X_test[:,:,:,np.newaxis], y_test, X_val[:,:,:,np.newaxis], y_val
	return X_train[:,np.newaxis,:,:], y_train, X_test[:,np.newaxis,:,:], y_test, X_val[:,np.newaxis,:,:], y_val

def write_to_hdf5(csv_file, output_file, num_lines = None, mirror = False, center=True):
	"""
	Function that takes as input a filname to a csv file that assumes the following formatting:
	emotion, pixels (2034 of them), usage (train, test, val)
	Returns the following numpy arrays:
	- X_train, y_train (respectively (N, 48, 48), (N,) representing raw grayscale pixels and emotion labels)
	- X_test, y_test
	- X_val, y_val
	"""

	# read in data using previously defined function into numpy arrays
	X_train, y_train, X_test, y_test, X_val, y_val = read_faces_csv(csv_file, num_lines, mirror, center)

	# write to hdf5
	with h5py.File(output_file, 'w') as hf:
		# create different groups separate train, test and validation sets
		train = hf.create_group('train')
		train.create_dataset('X_train', data = X_train)
		train.create_dataset('y_train', data = y_train)

		test = hf.create_group('test')
		test.create_dataset('X_test', data = X_test)
		test.create_dataset('y_test', data = y_test)

		val = hf.create_group('val')
		val.create_dataset('X_val', data = X_val)
		val.create_dataset('y_val', data = y_val)
