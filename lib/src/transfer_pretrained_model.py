from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import mnist

def keras_to_pyt(km, pm):
    weight_dict = dict()
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
    pyt_state_dict = pm.state_dict()
    for key in pyt_state_dict.keys():
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
    pm.load_state_dict(pyt_state_dict)
    return pm


def main():
    # define the model
    keras_network = keras_Net()
    keras_network.load_weights("models/mnist")
    # print_keras_model(keras_network)
    pytorch_network = pytorch_Net()

    # transfer keras model to pytorch
    pytorch_network = keras_to_pyt(keras_network, pytorch_network)
    torch.save(pytorch_network.state_dict(), "pyt_model.pt")
    ############### test the performance of two models#########
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    inp = np.expand_dims(x_train[0:100], axis=1)
    print('inp.shape', inp.shape)

    inp_pyt = torch.autograd.Variable(torch.from_numpy(inp.copy()).float())
    inp_keras = np.transpose(inp.copy(), (0, 2, 3, 1))
    pyt_res = pytorch_network(inp_pyt).data.numpy()
    keras_res = keras_network.predict(x=inp_keras, verbose=1)
    for i in range(100):
        predict1 = np.argmax(pyt_res[i])
        predict2 = np.argmax(keras_res[i])
        if predict1 != predict2:
            print("ERROR: Two model ooutput are different!")
        elif predict1 != y_train[i]:
            print("The model predict for {}th image is wrong".format(i+1))