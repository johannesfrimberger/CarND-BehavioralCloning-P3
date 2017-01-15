#Capturing input data from a joystick using Pygame

#Import pygame and init
from pygame import *
init()

#Setup and init joystick
j=joystick.Joystick(0)
j.init()

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
from keras.optimizers import SGD, Adam, RMSprop

from threading import Thread, Lock

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf
from model import process_image

import random

from time import sleep

sio = socketio.Server()
app = Flask(__name__)

lp_steering_angle = None
prev_image_array = None

training_images = []
training_steering = []

mutex = Lock()

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

    global lp_steering_angle

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    if auto_mode:
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    else:
        alpha = 0.3
        #print("Request {}".format(joy_steering))
        lp_steering_angle = (1.0 - alpha) * lp_steering_angle + alpha * joy_steering
        steering_angle = lp_steering_angle

        mutex.acquire()
        try:
            training_images.append(image_array)
            training_steering.append(steering_angle)
        finally:
            mutex.release()

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.0
    #print(steering_angle, throttle)
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


def read_inputs():
    # Check init status
    if j.get_init() == 1:
        print("Joystick is initialized")
    global auto_mode
    global joy_steering
    auto_mode = True
    joy_steering = 0
    # Setup event information and print data from joystick
    while 1:
        for e in event.get():
            if e.type == JOYAXISMOTION and e.dict["axis"] == 0:
                joy_steering = 0.8563391 * e.dict["value"]
            elif e.type == JOYBUTTONDOWN and e.dict["button"] == 0:
                auto_mode = not auto_mode
                print("Switch auto_mode to {}".format(auto_mode))
            elif e.type == JOYBUTTONDOWN and e.dict["button"] == 1:
                joy_steering = 0
                print("Reset steering request to {}".format(joy_steering))

def live_trainer():
    while 1:

        if len(training_images) > 10:

            mutex.acquire()
            try:
                img = np.asarray(training_images)
                st = np.asarray(training_steering)
                training_images[:] = []
                training_steering[:] = []
            finally:
                mutex.release()

            print(img.shape)
            print(st.shape)

            model.fit(img, st, batch_size=5, nb_epoch=2, verbose=1)


        sleep(5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')

    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        global model
        model = model_from_json(json.load(jfile))

    optimizer = Adam(lr=0.000001)

    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    model.compile(optimizer=optimizer, loss="mse")

    # Start separate thread to read inputs
    thread = Thread(target=read_inputs)
    thread.start()

    # Start separate thread to read inputs
    thread2 = Thread(target=live_trainer)
    thread2.start()

    # Initialize global variable for
    lp_steering_angle = 0

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)