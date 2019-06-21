import os

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.optimizers import Adam

from game.game import ACTIONS
from model.image_processing import img_rows, img_cols
from model.model_params import img_channels, LEARNING_RATE
from paths import scores_file_path, basepath

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


def buildmodel():
    model = Sequential()
    model.add(
        Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)

    # create model file if not present
    if not os.path.isfile('model.h5'):
        model.save_weights('model.h5')
    print(model.summary())
    return model
