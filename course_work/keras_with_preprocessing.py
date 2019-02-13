import keras
from keras.applications.vgg16 import VGG16
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.morphology as morp
from keras.datasets import cifar10
from skimage.filters import rank
from sklearn.utils import shuffle
import cv2
import os
import csv
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.applications import vgg16
from keras import backend as K

import course_work.preprocess_image as preproc


def list_images(dataset, dataset_y, ylabel="", cmap=None):
    """
        Display a list of images in a single figure with matplotlib.
            Parameters:
                images: An np.array compatible with plt.imshow.
                lanel (Default = No label): A string to be used as a label for each image.
                cmap (Default = None): Used to display gray images.
        """
    plt.figure(figsize=(15, 16))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        indx = random.randint(0, len(dataset))
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(dataset[indx].shape) == 2 else cmap
        plt.imshow(dataset[indx], cmap=cmap)
        plt.xlabel(signs[dataset_y[indx]])
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


training_file = "./traffic-signs-data/train.p"
validation_file = "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

signs = []
with open('signnames.csv', 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames, None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples
n_train = X_train.shape[0]

# Number of testing examples
n_test = X_test.shape[0]

# Number of validation examples.
n_validation = X_valid.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

X_train, y_train = shuffle(X_train, y_train, random_state=0)
# print("gray_scale")
# X_train = list(map(preproc.adaptive_mean, X_train))
X_train = list(map(preproc.gray_scale, X_train))
X_train = preproc.image_normalize(X_train)

list_images(X_train, y_train, "gray_scale, adaptive_mean")

model = keras.applications.VGG19(include_top=False, input_shape=(32, 32, 3))
# model = keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
flatten = Flatten()
new_layer2 = Dense(n_classes, activation='softmax', name='my_dense_2')
inp2 = model.input
out2 = new_layer2(flatten(model.output))
resultModel = Model(inp2, out2)
resultModel.summary(line_length=150)

batch_size = 64
epochs = 50

resultModel.compile(optimizer="adam",
                    loss=keras.losses.categorical_crossentropy,
                    metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)
resultModel.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_valid, y_valid)
)
resultModel.save_weights('bottleneck_fc_model.h5')
