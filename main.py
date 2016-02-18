import numpy as np
import matplotlib.pyplot as plt
import process_data

from cs231n.classifiers.cnn import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

import sys
import yaml

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Analyzing command line arguments
if len(sys.argv) < 3:
  print 'Usage:'
  print '  python %s <.csv file> <.yaml config file>' % sys.argv[0]
  exit()

csv_file, config_file = sys.argv[1], sys.argv[2]

# reading in configuration file with parameters
f = open(config_file)
options = yaml.load(f)
f.close()

num_train = options['num_train']
num_val = options['num_val']
num_filters, filter_size = options['filters'], options['filter_size']
weight = float(options['weight_scale'])
hidden_dim = options['hidden_dim']
reg = float(options['reg'])
num_epochs = options['num_epochs']
batch_size = options['batch_size']
learning_rate = float(options['learning_rate'])
lr_decay = float(options['lr_decay'])

# read in data
X_train, y_train, X_test, y_test, X_val, y_val = process_data.read_faces_csv(csv_file)

N_train, _, H, W = X_train.shape
N_val, _, _, _ = X_val.shape

data = {'X_train' : X_train[:num_train,:,:,:], 'y_train' : y_train[:num_train], \
		"X_val" : X_val[:num_val,:,:,:], "y_val" : y_val[:num_val] }

print "Done loading data"

# reg = 0.000511417045343
# learning_rate = 0.000135139484703
# weight_scale = 0.002088691359

model = ThreeLayerConvNet(input_dim=(1, H, W), num_classes=7, num_filters=num_filters, filter_size=filter_size, \
							weight_scale=weight, hidden_dim=hidden_dim, reg=reg)

solver = Solver(model, data,
                num_epochs=num_epochs, batch_size=batch_size,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=lr_decay,
                verbose=True, print_every=10)

solver.train()

def getConfusionMatrix(y_pred, y_true, numClasses=7, asFraction = True):
    """
    Returns confusion matrix.
    Row: True Value
    Column: Prediction
    Entries: Counts
    asFraction: if False return counts, otherwise fraction
    """
    confusionMatrix = np.zeros((numClasses,numClasses), np.int)
    for i in range(y_pred.shape[0]):
        confusionMatrix[y_true[i], y_pred[i]] += 1
    if asFraction:
        rowSums = confusionMatrix.sum(axis=1)
        return confusionMatrix.astype(float)/rowSums[:, np.newaxis]
    else:
        return confusionMatrix


scores = solver.model.loss(solver.X_val)
y_pred = np.argmax(scores, axis=1)
print getConfusionMatrix(y_pred, solver.y_val)
