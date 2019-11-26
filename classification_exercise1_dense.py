import numpy as np
# import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(len(x_train))
print(x_test.shape)

x_train = x_train.reshape(len(x_train), -1)
x_test = x_test.reshape(len(x_test), -1)
print(x_train)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - 127) / 127
x_test = (x_test - 127) / 127
print(x_train)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, input_shape=(784,), kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(512, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1, validation_split=0.05)

loss,accuracy = model.evaluate(x_test,y_test)
print("testloss:",loss)
print("testaccuracy:",accuracy)