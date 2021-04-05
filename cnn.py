import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
import os
import random
import numpy as np
import pickle


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        counter = CATEGORIES.index(category)  # TRACK COUNTER INDEX
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # DON'T NEED COLOR

                scaled = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([scaled, counter])  # LIST OF TUPLES
                # EACH TUPLE IS THE SCALED IMG DATA AND THE INTEGER ASSOCIATED WITH THE CATEGORY OF WASTE
            except Exception as e:
                pass


IMG_SIZE = 75
# create the data set
ORIGDIR = os.getcwd()
DATADIR = os.path.join(ORIGDIR, 'TRAIN')  # MOVE INTO THE TRAIN FOLDER
CATEGORIES = os.listdir('TRAIN')
outputs = len(CATEGORIES)
if not os.path.exists('X_train.pkl') or not os.path.exists('y_train.pkl'):
    training_data = []
    create_training_data()
    random.shuffle(training_data)
    X_train = []
    y_train = []

    for feature, label in training_data:
        X_train.append(feature)  # EXTRACT ARRAYS FROM LIST OF TUPLES
        y_train.append(label)  # EXTRACT WASTE CATEGORY INTEGER/INDEX FROM LIST OF TUPLES

    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_train = np.true_divide(X_train, 255)
    y_train = np.array(y_train)
    os.chdir(ORIGDIR)
    with open('X_train.pkl', 'wb') as file:
        pickle.dump(X_train, file)
    with open('y_train.pkl', 'wb') as file:
        pickle.dump(y_train, file)

else:
    with open('X_train.pkl', 'rb') as file:
        X_train = pickle.load(file)

    with open('y_train.pkl', 'rb') as file:
        y_train = pickle.load(file)

# tf.config.list_physical_devices('GPU')

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    #MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(units=outputs, activation='softmax'),
])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, verbose=2)
model.save('classified_waste_color.model')
