from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import numpy as np
import pandas as pd
import os
import sys

csv = pd.read_csv('/home/mike/Documents/test.csv')
test_data_x = np.array(csv)

print(test_data_x)

model = Sequential()
model.add(Dense(units=15,
                input_dim=10,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='relu'))

model.add(Dense(units=13,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='softmax'))

model.load_weights("/home/mike/Documents/weights-best.h5")
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
prediction=model.predict_classes(-test_data_x)

print("\nLocation:", prediction)
