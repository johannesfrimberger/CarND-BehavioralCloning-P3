Overview
--------

This repository contains my solution for the Behavioral Cloning
Project of the Udacity Self-Driving Car Nanodegree.

It shows how to set up and train a convolutional neural network
that is capable of learning how to drive a car.

All Training and testing took place with Ubuntu 16.04 and the
beta version of the Udacity Self-Driving Car Simulator.

Outline
-------

1. Solution Design Process
2. Model Architecture
3. Training
4. Live Trainer
5. Usage
6. Summary
7. References

Solution Design Process
-----------------------



Model Architecture
------------------


Training
--------

Initially the model was trained using the dataset provided by Udacity.
I used an Adam Optimizer with a decreased learning rate of 0.00001
compared to the default settings.

Training was done on 5 epochs of the training data with only 5% of the
data used for validation. This low number was chosen as the metric (MSE)
does not really provide a good measurement of the how the model will
perform in the simulator.

With this training the car was able to drive to the first left turn but
could not take it.

Live Trainer
------------

<img src="img/controller.jpg" width="200" height="200" /> <br />

Training with the Udacity dataset already gave a first
Compared to the solutions


<img src="img/360_Left_Stick.png" width="20" height="20" />
&nbsp;&nbsp;The left st <br />

<img src="img/360_A.png" width="20" height="20" />
&nbsp;&nbsp;The left st <br />

<img src="img/360_X.png" width="20" height="20" />
&nbsp;&nbsp;The left st <br />

<img src="img/360_B.png" width="20" height="20" />
&nbsp;&nbsp;The left st <br />


How to run the code
-------------------

To run the

For training the
```
python3 model.py -t data
```

The training has several options.
```
python3 model.py -t data -n 5 -a 0 -l 0
```

Start the
```
python3 drive.py model.json
```

Start the live trainer with model save under model.json. This model will
also be updated.
```
python3 live_trainer.py model.json
```

Summary
-------

In the end the model is capable of providing a

References
----------

[1] http://comma.ai <br />
[2] https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf <br />
[3] https://upload.wikimedia.org/wikipedia/commons/4/4d/Xbox-360-Wireless-Controller-White.jpg <br />
[4] http://opengameart.org/content/free-keyboard-and-controllers-prompts-pack <br />