import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.backend import stop_gradient

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import argparse
import csv
import cv2
import os
import re
import math
import json
import random

def create_cnn_model(feature_shape):

    p_dropout = 0.2
    pool_size = (2, 3)
    model = Sequential()

    model.add(MaxPooling2D(pool_size=pool_size, input_shape=feature_shape))
    model.add(Lambda(lambda x: x / 127.5 - 1.))

    model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))

    model.add(Flatten())
    model.add(Dropout(p_dropout))

    model.add(ELU())
    model.add(Dense(1164))
    model.add(Dropout(p_dropout))
    model.add(ELU())

    model.add(Dense(100))
    model.add(Dropout(p_dropout))
    model.add(ELU())

    model.add(Dense(50))
    model.add(Dropout(p_dropout))
    model.add(ELU())

    model.add(Dense(10))
    model.add(Dropout(p_dropout))
    model.add(ELU())

    model.add(Dense(1))

    return model

def clean_up_image_path(filename, root):
    """
    Check if root is in filename and add it if not
    :param filename:
    :param root:
    :return:
    """

    if root in filename:
        output = re.sub(r'.*' + root, root, filename)
    else:
        output = os.path.join(root, filename)

    return output.replace(" ", "")

def merge_training_data(folder):
    """
    Look for all driving_log.csv files in given folder and merge
    :param folder: Folder to search for driving_log.csv files
    :return: Np array with ["image_center", "image_left", "image_right"] and
    one with the corresponding steering angle
    """

    merged_features = np.zeros([0, 3], dtype='a5')
    merged_steering = np.zeros([0])

    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith("driving_log.csv"):
                with open(os.path.join(root, file), 'r') as f:
                    reader = csv.reader(f)
                    all_data = list(reader)[1:]

                    # Modify path to image for each element
                    for ind in range(0, len(all_data)):
                        scene = all_data[ind]
                        scene[0:3] = [clean_up_image_path(scene[i], root) for i in range(0, 3)]
                        all_data[ind] = scene

                    features = np.asarray(all_data)[:, 0:3]
                    steering = np.asarray(all_data)[:, 3]
                    steering = steering.astype(np.float)
                    merged_features = np.concatenate((merged_features, features), axis=0)
                    merged_steering = np.concatenate((merged_steering, steering), axis=0)

    return [merged_features, merged_steering]

def apply_roi(img):
    """

    :param img:
    :return:
    """
    img = img[60:140, 40:280]
    return cv2.resize(img, (200, 66))

def process_image(colorImage):
    """
    Load image and convert it to YUV color space
    :param filename: Full path to image file
    :return: Np array containing image
    """
    yuv = cv2.cvtColor(colorImage, cv2.COLOR_RGB2YUV)
    return apply_roi(yuv).astype(np.float)


def process_inputs(feature, steering, steering_offset = 0.):
    """

    :param feature:
    :param steering:
    :param steering_fac:
    :return:
    """

    data = []
    for ind, filename in enumerate(feature):
        colorImage = mpimg.imread(filename)
        data.append(process_image(colorImage))

    return np.asarray(data), (steering + steering_offset)


def batch_generator(features, steering, batch_size, use_left_right=False):
    """
    Generator returning a batch of data for every iteration
    :param features: Np array of image locations
    :param steering: Np arrayof steering requests
    :param batch_size: number of batches in an iteration
    :return: tuple of features and
    """
    # Get number of features
    n_features = features.shape[0]

    # Set start index to 0
    start_ind = 0

    while 1:

        current_features = features[start_ind:(start_ind + batch_size)]
        current_steering = steering[start_ind:(start_ind + batch_size)]

        (features1, steering1) = process_inputs(current_features[:, 0], current_steering, steering_offset=0.)
        (features2, steering2) = process_inputs(current_features[:, 1], current_steering, steering_offset=0.1)
        (features3, steering3) = process_inputs(current_features[:, 2], current_steering, steering_offset=-0.1)

        if use_left_right:
            batch_features = np.concatenate((features1, features2, features3), axis=0)
            batch_steering = np.concatenate((steering1, steering2, steering3), axis=0)
        else:
            batch_features = features1
            batch_steering = steering1

        start_ind += batch_size

        # Reset start index if number of features is exceeded
        if start_ind >= n_features:
            start_ind = 0

        yield(batch_features, batch_steering)

def main(training_data, n_epochs, load_model):

    # Set shape of input images
    feature_shape = (66, 200, 3)

    # Load all training data from data folder
    print("Read training data from {}".format(training_data))
    all_features, all_steering = merge_training_data(training_data)

    # Split data into training and validation set.
    # A test set is not required as the performance is evaluated on the track
    training_features, valid_features, training_steering, valid_steering = train_test_split(
        all_features,
        all_steering,
        test_size=0.05,
        random_state=10)

    steering_thresh = 0.03

    steering_left = np.where(all_steering < -steering_thresh)
    steering_right = np.where(all_steering > steering_thresh)
    steering_center = np.where(np.logical_and(all_steering > -steering_thresh, all_steering < steering_thresh))

    print(steering_left)

    random.seed(10)
    random.shuffle(steering_left)
    random.shuffle(steering_right)
    random.shuffle(steering_center)

    n_left = steering_left[0].shape[0]
    n_right = steering_right[0].shape[0]
    n_center = steering_center[0].shape[0]

    print(n_left)
    print(n_right)
    print(n_center)

    # Give a short summary of the data
    n_training_scene = training_features.shape[0]
    n_training_features = n_training_scene * 3
    print("Training the model on {} images from {} scenes".format(n_training_features, n_training_scene))
    n_valid_scenes = valid_features.shape[0]
    n_valid_features = n_valid_scenes * 3
    print("Validation of the model is done on {} images from {} scenes".format(n_valid_features, n_valid_scenes))

    # Define model
    if load_model:
        filename = "model.json"
        print("Load Model {}".format(filename))
        with open(filename, 'r') as jfile:
            model = model_from_json(json.load(jfile))

        weights_file = filename.replace('json', 'h5')
        model.load_weights(weights_file)
        #optimizer = Adam(lr=0.00001, decay=0.01)
        optimizer = Adam(lr=0.00001)

    else:
        print("Create new model")
        model = create_cnn_model(feature_shape)
        #optimizer = Adam(lr=0.00001, decay=0.0)
        optimizer = Adam(lr=0.00001)

    model.compile(optimizer=optimizer, loss="mse")

    # Define some parameters for optimization
    batch_size = 5

    training_samples_per_epoch = math.ceil(n_training_scene / batch_size)
    valid_samples_per_epoch = math.ceil(n_valid_features / batch_size)

    history = model.fit_generator(batch_generator(training_features, training_steering, batch_size, False),
                                  samples_per_epoch=training_samples_per_epoch, nb_epoch=n_epochs,
                                  validation_data=batch_generator(valid_features, valid_steering, batch_size),
                                  nb_val_samples=valid_samples_per_epoch, verbose=1)

    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

    model.save_weights("model.h5")

    # Predict steering angle for small subsample to check basis functionality

    # Load all training data from data folder
    print("Read validation data from {}".format("data/test"))
    all_features, all_steering = merge_training_data("data/test")

    # Get first batch of features
    b = batch_generator(all_features, all_steering, batch_size)
    inputFeature, inputSteering = b.__next__()

    prediction = model.predict(inputFeature)

    center = prediction[0][0]
    left = prediction[1][0]
    right = prediction[2][0]

    print("Center is {} ({})".format(center, inputSteering[0]))
    print("Left is {} ({})".format(left, inputSteering[1]))
    print("Right is {} ({})".format(right, inputSteering[2]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build Remote Driving Model')
    parser.add_argument('training_data', type=str,
                        help='Path to training data used to train and validate classifier.')
    parser.add_argument('n_epochs', type=int,
                       help='Path to training data used to train and validate classifier.')
    parser.add_argument('load_model', type=int,
                        help='Load model')
    args = parser.parse_args()

    main(args.training_data, args.n_epochs, (args.load_model > 0))
