import numpy as np
import matplotlib.pyplot as plt
import process_data

from cs231n.classifiers.cnn import *
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

X_train, y_train, X_test, y_test, X_val, y_val = process_data.read_faces_csv('fer2013/fer2013.csv')

N_train, img_size, _ = X_train.shape
N_val, _, _ = X_val.shape

X_train = X_train.reshape(N_train, 1, img_size, img_size) # fake extra dim for channels
X_val = X_val.reshape(N_val, 1, img_size, img_size)


data = {'X_train' : X_train[:5000,:,:,:], 'y_train' : y_train[:5000], "X_val" : X_val, "y_val" : y_val }

reg = 0.000511417045343
learning_rate = 0.000135139484703
weight_scale = 0.002088691359

model = ThreeLayerConvNet(input_dim=(1, img_size, img_size), num_classes=7, num_filters=32, filter_size=5, weight_scale=weight_scale, hidden_dim=500, reg=reg)

solver = Solver(model, data,
                num_epochs=5, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': learning_rate,
                },
                lr_decay=0.95,
                verbose=True, print_every=10)

solver.train()
