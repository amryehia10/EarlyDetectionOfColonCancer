from keras.applications.vgg16 import VGG16
import tensorflow as tf
import keras.metrics as t
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation, Conv2D, MaxPool2D
from keras.models import Sequential, load_model


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3), classes=2)
model = Sequential()
model.add(base_model)
model.add(BatchNormalization()) #normalize inputs to each layer which can speed up training
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1024, activation=("relu")))
model.add(Dense(2, activation=("sigmoid")))

# Model summary
model.summary()