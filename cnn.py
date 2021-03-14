import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
import os
import numpy as np
import pickle


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        counter = CATEGORIES.index(category)  # TRACK COUNTER INDEX
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # DON'T NEED COLOR
                scaled = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([scaled, counter])  # LIST OF ARRAYS
            except Exception as e:
                pass


IMG_SIZE = 50
CATEGORIES = ['O', 'R']
if not os.path.exists('X.pkl'):
    # create the data set
    ORIGDIR = os.getcwd()
    DATADIR = os.path.join(ORIGDIR, 'TRAIN')  # MOVE INTO THE TRAIN FOLDER
    os.chdir(DATADIR)
    training_data = []
    create_training_data()

    X = []
    y = []

    for feature, label in training_data:  # EXTRACT ARRAYS FROM LIST OF ARRAYS AND CONVERT INTO NUMPY ARRAY
        X.append(feature)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    os.chdir(ORIGDIR)
    with open('X.pkl', 'wb') as file:
        pickle.dump(X, file)
    with open('y.pkl', 'wb') as file:
        pickle.dump(y, file)

with open('X.pkl', 'rb') as file:
    X = pickle.load(file)
    X = np.true_divide(X, 255)
with open('y.pkl', 'rb') as file:
    y = pickle.load(file)

# tf.config.list_physical_devices('GPU')

model = keras.Sequential([
                   Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(X.shape[1:])),
                   MaxPooling2D(pool_size=(2, 2), strides=2),
                   Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                   MaxPooling2D(pool_size=(2, 2), strides=2),
                   Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                   MaxPooling2D(pool_size=(2, 2), strides=2),
                   Flatten(),
                   Dense(units=1, activation='softmax')
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=10, verbose=2)
