import os
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.utils import np_utils

csv = pd.read_csv('location25_train.csv')
csv = np.array(csv)

train_data = -np.hsplit(csv, (0, 25))[1]
train_label = np.hsplit(csv, (0, 25))[2]
train_label_OneHot = np_utils.to_categorical(train_label)

model = Sequential()
model.add(Dense(input_dim = 25,
                units = 120,
                kernel_initializer = 'normal',
                bias_initializer = 'zeros',
                activation = 'relu'))
model.add(Dense(units = 120,
                kernel_initializer = 'normal',
                bias_initializer = 'zeros',
                activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

filepath = 'Save'
if not os.path.isdir(filepath):
    os.mkdir(filepath)

checkpoint = ModelCheckpoint(filepath + "/weights-best.h5", monitor = 'val_loss', save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]

model.fit(train_data, train_label_OneHot, batch_size = 200, epochs = 100000, callbacks = callbacks_list, validation_data = (train_data, train_label_OneHot))
