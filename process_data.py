import csv
import numpy as np

csv_file = open('../fer2013/fer2013.csv')

reader = csv.reader(csv_file)

# Discard header
row = next(reader)

y_list = []
X_list = []

for row in reader:
	y_str, X_row_str = (row[0], row[1])
	y = int(y_str)
	y_list.append([y])

	X_row_strs = X_row_str.split(' ')
	X_row = [float(x) for x in X_row_strs]
	X_list.append(X_row)

X = np.asarray(X_list).astype('float64')

y = np.asarray(y_list)