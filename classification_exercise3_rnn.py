import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Activation, SimpleRNN, Flatten, Input, LSTM, GRU, Reshape
from keras.optimizers import Adam

# download the mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# print((X_test, Y_test))
X_train = X_train.reshape(-1, 28, 28)/255
X_test = X_test.reshape(-1, 28, 28)/255
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)
# print(X_train.shape)
# print(Y_train.shape)
inputs = Input(shape=(28, 28))
# inputs = Input(shape=(28, 28, 1))

# x = Reshape(input_shape=(28, 28, 1), target_shape=(28, 28*1))(inputs)
# x = LSTM(units=32, input_shape=(28, 28), return_sequences=False)(x)


x = SimpleRNN(64, unroll=True)(inputs)
x = Dense(10)(x)
x = Activation('softmax')(x)

model = Model(input=inputs, output=x)
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

BATCH_INDEX=0
BATCH_SIZE = 50

for step in range(10001):
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = Y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE,:]
    # print(Y_batch.shape)
    loss= model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    # print('loss, accuracy: ', (loss, accuracy))
    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
