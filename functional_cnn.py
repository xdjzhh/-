from keras.layers import Conv2D, BatchNormalization, Input, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Activation, Dense, Concatenate, Add
from keras.models import Model
import keras.backend as K
from keras.utils.vis_utils import plot_model


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255
X_test = X_test.reshape(-1, 28, 28, 1) / 255
Y_train = to_categorical(Y_train, num_classes=10)
Y_test = to_categorical(Y_test, num_classes=10)


def basic_nn_block(x):
    init = x

    x = Conv2D(filters=32, kernel_size=(3, 3), use_bias=False, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(1, 1), use_bias=False, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False, strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    if K.shape(init)[-1] != 64:
        init = Conv2D(filters=64, kernel_size=(1, 1), use_bias=False, strides=(1, 1), padding='same')(init)
    x = Add()([x, init])

    return x


def define_model():

    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), use_bias=False)(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)

    # x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=False)(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    #
    # x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=False)(x)
    # x1 = BatchNormalization(axis=-1)(x1)
    # x1 = Activation('relu')(x1)
    #
    # x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), use_bias=False)(x)
    # x2 = BatchNormalization(axis=-1)(x2)
    # x2 = Activation('relu')(x2)
    #
    # x = Concatenate(axis=-1)([x1, x2])

    x = basic_nn_block(x)
    x = basic_nn_block(x)
    x = basic_nn_block(x)

    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


_model = define_model()
print(_model.summary())
plot_model(_model, show_shapes=True)
# _model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10)

