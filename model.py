import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D
from keras.optimizers import Adam

import argparse
import csv
import cv2
import os
import re
import json


def create_cnn_model(feature_shape):
    """
    Create a convolutional neural network in Keras to predict steering angle from image inputs.
    This model is inspired by the one used in the http://comma.ai/ project.
    :param feature_shape: Shape of the features (images) in (ch, col, row) format
    :return: Keras CNN model
    """
    # Decode feature shape
    ch, col, row = feature_shape

    # Init Keras model
    model = Sequential()

    # Add normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=(ch, row, col),
                     output_shape=(ch, row, col)))

    # Add 1st convolutional layer with 16 8x8 kernels and
    # exponential linear unit applied to output
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())

    # Add 2nd convolutional layer with 32 5x5 kernels and
    # exponential linear unit applied to output
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())

    # Add 3rd convolutional layer with 64 5x5 kernels and flatten output
    # exponential linear unit applied to output and dropout applied to flatten layer
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())

    # Add a single dense layer with 512 neurons
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())

    # Add output layer for steering angle
    model.add(Dense(1))

    return model


def clean_up_image_path(filename, root):
    """
    Convert filename such that it contains relative path to image
    :param filename: Filename that either contains relative or absolute path
    :param root: Root folder of the python process
    :return: Realtive filename
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
    Apply fixed roi to an image and convert it to 66x200 format
    :param img: Input image
    :return: ROI of this image
    """
    img = img[60:140, 40:280]
    data = cv2.resize(img, (200, 66))
    return np.swapaxes(data, 0, 2)


def process_image(color_image):
    """
    Convert image to YUV color space, add red channel and apply ROI
    :param color_image: RGB image that should be processed
    :return: Np array containing feature
    """
    yuv = cv2.cvtColor(color_image, cv2.COLOR_RGB2YUV)
    r_channel = color_image[:, :, 0]
    r_channel = r_channel[:, :, np.newaxis]
    data = np.concatenate((yuv, r_channel), axis=2)

    return apply_roi(data).astype(np.float)


def process_inputs(feature, steering, steering_offset=0.):
    """
    Take a list of file names, read the corresponding images and pre-process them
    :param feature: List of file names that should be processed
    :param steering: List of steering angles (same length as feature list)
    :param steering_offset: Fixed offset that is added to steering angles
    :return: tuple of features and steering angle
    """
    data = []
    for filename in feature:
        color_image = mpimg.imread(filename)
        data.append(process_image(color_image))

    return np.asarray(data), (steering + steering_offset)


def batch_generator(features, steering, batch_size, use_left_right=False):
    """
    Generator returning a batch of data for every iteration
    :param features: Np array of image locations
    :param steering: Np array of steering requests
    :param batch_size: number of batches in an iteration
    :param use_left_right: Use left and right images for training
    :return: tuple of features and steering angle
    """
    # Get number of features
    n_features = features.shape[0]

    # Set start index to 0
    ind = 0

    while 1:
        current_features = features[ind:(ind + batch_size)]
        current_steering = steering[ind:(ind + batch_size)]

        (features1, steering1) = process_inputs(current_features[:, 0], current_steering)

        # If additional data should be used add left and right images with steering offset
        if use_left_right:
            (features2, steering2) = process_inputs(current_features[:, 1], current_steering, steering_offset=-0.02)
            (features3, steering3) = process_inputs(current_features[:, 2], current_steering, steering_offset=0.02)

            batch_features = np.concatenate((features1, features2, features3), axis=0)
            batch_steering = np.concatenate((steering1, steering2, steering3), axis=0)
        else:
            batch_features = features1
            batch_steering = steering1

        # Increase index
            ind += batch_size

        # Reset index if number of features is exceeded
        if ind >= n_features:
            ind = 0

        # Return batch
        yield(batch_features, batch_steering)


def main(training_data, n_epochs, load_model, additional_data):
    """
    Train a convolutional neural network to predict the steering angle
    from images and save this as model.json
    :param training_data: Folder where training images are stored
    :param n_epochs: Number of epochs used for training
    :param load_model: Load model from json file or create new model
    :param additional_data: Additionally use left and right images
    """
    # Set shape of input images (for convenience)
    feature_shape = (4, 66, 200)

    model_filename = "model.json"
    weights_filename = model_filename.replace('json', 'h5')

    # Load all training data from data folder
    all_features, all_steering = merge_training_data(training_data)

    # Split data into training and validation set.
    # A test set is not required as the performance is evaluated on the track
    training_features, valid_features, training_steering, valid_steering = train_test_split(
        all_features,
        all_steering,
        test_size=0.05,
        random_state=10)

    # Give a short summary of the data
    n_training_scene = training_features.shape[0]
    print("Training the model on {} scenes".format(n_training_scene))
    n_valid_scenes = valid_features.shape[0]
    print("Validation of the model is done on {} scenes".format(n_valid_scenes))

    # Define model
    if load_model:
        print("Load Model {}".format(model_filename))
        with open(model_filename, 'r') as jfile:
            model = model_from_json(json.load(jfile))
        model.load_weights(weights_filename)
        optimizer = Adam(lr=0.000001)
    else:
        print("Create new model")
        model = create_cnn_model(feature_shape)
        optimizer = Adam(lr=0.00001)

    model.compile(optimizer=optimizer, loss="mse")

    # Define parameters for optimization
    batch_size = 10

    # Using left and right images gives 3 times the number of samples
    if additional_data:
        training_samples_per_epoch = n_training_scene * 3
    else:
        training_samples_per_epoch = n_training_scene

    valid_samples_per_epoch = n_valid_scenes

    model.fit_generator(batch_generator(training_features, training_steering, batch_size, additional_data),
                        samples_per_epoch=training_samples_per_epoch, nb_epoch=n_epochs,
                        validation_data=batch_generator(valid_features, valid_steering, batch_size),
                        nb_val_samples=valid_samples_per_epoch, verbose=1)

    with open(model_filename, "w") as outfile:
        json.dump(model.to_json(), outfile)

    model.save_weights(weights_filename)

    # Predict steering angle for small subsample to check basis functionality

    # Load all training data from data folder
    print("Read validation data from {}".format("data/test"))
    all_features, all_steering = merge_training_data("data/test")

    # Get first batch of features
    b = batch_generator(all_features, all_steering, batch_size)
    input_feature, input_steering = b.__next__()

    prediction = model.predict(input_feature)

    center = prediction[0][0]
    left = prediction[1][0]
    right = prediction[2][0]

    print("Prediction for center test image is {} ({})".format(center, input_steering[0]))
    print("Prediction for left test image is {} ({})".format(left, input_steering[1]))
    print("Prediction for right test image is {} ({})".format(right, input_steering[2]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Build Remote Driving Model')

    parser.add_argument("-t", "--training_data", help="Path to training data used to train and validate classifier",
                        required=True)
    parser.add_argument("-n", "--n_epochs", help="Number of epochs used for training", type=int,
                        default=5)
    parser.add_argument("-l", "--load_model", help="Load model and improve results", type=int,
                        default=0)
    parser.add_argument("-a", "--additional_data", help="Use left and right images for training", type=int,
                        default=0)
    args = parser.parse_args()

    # Print current settings
    print("Settings")
    print("training_data: {}".format(args.training_data))
    print("n_epochs: {}".format(args.n_epochs))
    print("load_model: {}".format(args.load_model))
    print("additional_data: {}".format(args.additional_data))
    print("Start training model")

    # Run main method
    main(args.training_data, args.n_epochs, (args.load_model > 0), (args.additional_data > 0))
