import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import Adam

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
from model import process_image

#Import pygame and init
from pygame import *
init()

import random

#Setup and init joystick
j = joystick.Joystick(0)
j.init()

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

lp_steering_angle = 0
auto_mode = True

training_images = []
training_steering = []

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = process_image(np.asarray(image))
    transformed_image_array = image_array[None, :, :, :]

    # Define global variables
    global lp_steering_angle
    global auto_mode

    # Init variables and copy previous state
    joy_steering = 0.
    prev_auto_mode = auto_mode

    # Read data from gamepad
    for e in event.get():
        # Read left joystick data in left/right direction
        if e.type == JOYAXISMOTION and e.dict["axis"] == 0:
            # Value has a range of -1 to 1 --> Multiply with maximum steering angle
            joy_steering = 0.8563391 * e.dict["value"]
        # Toggle auto_mode for A and X Button
        elif e.type == JOYBUTTONDOWN and (e.dict["button"] == 0 or e.dict["button"] == 2):
            auto_mode = not auto_mode
            print("Switch auto_mode to {}".format(auto_mode))
            # For X button additionally remove previously stored training data (no training requested)
            if e.dict["button"] == 2:
                training_images[:] = []
                training_steering[:] = []
        # For B button reset steering low pass to 0
        elif e.type == JOYBUTTONDOWN and e.dict["button"] == 1:
            lp_steering_angle = 0
            print("Reset steering request to {}".format(joy_steering))

    # Switch between autonomous and manual driving
    if auto_mode:
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    else:
        # Low Pass joystick input
        alpha = 0.3
        lp_steering_angle = (1.0 - alpha) * lp_steering_angle + alpha * joy_steering
        steering_angle = lp_steering_angle

        # Add images and steering angle to array
        training_images.append(image_array)
        training_steering.append(steering_angle)

    # Train images if we switch from manual to automatic driving
    if auto_mode and not prev_auto_mode:
        # Check if there is data stored for training
        if len(training_images) > 0:
            print("Start training")

            # Shuffle data
            n_samples = len(training_images)
            ind = np.arange(n_samples)
            random.shuffle(ind)

            # Convert data to np arrays
            img = np.asarray(training_images)
            st = np.asarray(training_steering)

            # Remove data after copying it
            training_images[:] = []
            training_steering[:] = []

            # Start training
            model.fit(img[ind], st[ind], batch_size=10, nb_epoch=2, verbose=1)

            # Save model
            model_filename = "updated_model.json"
            weights_filename = model_filename.replace('json', 'h5')
            with open(model_filename, "w") as outfile:
                json.dump(model.to_json(), outfile)
            model.save_weights(weights_filename)

    # Set default throttle
    throttle = 0.5
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    optimizer = Adam(lr=0.00001)
    model.compile(optimizer=optimizer, loss="mse")

    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)