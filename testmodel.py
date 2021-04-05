import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import cv2
import os
import numpy as np
import pickle


def create_test_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        counter = CATEGORIES.index(category)  # TRACK COUNTER INDEX
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)  # DON'T NEED COLOR
                scaled = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([scaled, counter])  # LIST OF ARRAYS
            except Exception as e:
                pass


model = keras.models.load_model('classified_waste_color.model')
IMG_SIZE = 75
CATEGORIES = os.listdir('TEST')
if not os.path.exists('X_test.pkl') or not os.path.exists('y_test.pkl'):
    # create the data set
    ORIGDIR = os.getcwd()
    DATADIR = os.path.join(ORIGDIR, 'TEST')  # MOVE INTO THE TRAIN FOLDER
    os.chdir(DATADIR)
    test_data = []
    create_test_data()
    random.shuffle(test_data)
    X_test = []
    y_test = []

    for feature, label in test_data:  # EXTRACT ARRAYS FROM LIST OF ARRAYS AND CONVERT INTO NUMPY ARRAY
        X_test.append(feature)
        y_test.append(label)

    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_test = np.true_divide(X_test, 255)
    y_test = np.array(y_test)
    os.chdir(ORIGDIR)
    with open('X_test.pkl', 'wb') as file:
        pickle.dump(X_test, file)
    with open('y_test.pkl', 'wb') as file:
        pickle.dump(y_test, file)

else:
    with open('X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)

    with open('y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

# tf.config.list_physical_devices('GPU')

results = model.evaluate(X_test, y_test, batch_size=16)
print("test loss, test acc:", results)
