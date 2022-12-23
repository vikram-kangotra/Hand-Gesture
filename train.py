import os
import csv
import string
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras import layers
from sklearn.preprocessing import OneHotEncoder
import random

def get_data_from_csv(csv_file):
    label = csv_file.split('/')[-1].split('.')[0]
    dataset = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            dataset.append([row, label])
    return dataset


def get_data(base):
    dataset = []
    for file in os.listdir(base):
        dataset += get_data_from_csv(os.path.join(base, file))
    random.shuffle(dataset)
    return dataset


def get_data_and_labels(dataset):
    data = []
    labels = []
    for d in dataset:
        data.append(d[0])
        labels.append(d[1])
    return data, labels

def get_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


train_base = "extract/train"
test_base = "extract/test"

train_data = get_data(train_base)
test_data = get_data(test_base)

train_data, train_labels = get_data_and_labels(train_data)
test_data, test_labels = get_data_and_labels(test_data)

label_set = np.unique(train_labels)
input_shape = len(train_data[0])

num_classes = len(label_set)

x_train = np.array(train_data)
x_test = np.array(test_data)

y_train = OneHotEncoder().fit_transform(np.array(train_labels).reshape(-1, 1)).toarray()
y_test = OneHotEncoder().fit_transform(np.array(test_labels).reshape(-1, 1)).toarray()

model = get_model(input_shape, num_classes)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

model.save('model.h5')
