Overview
--------




Solution Design Proces
----------------------



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

<img src="img/controller.jpg" width="200" height="200" />



Usage
-----

The

Summary
-------

In the end the model is capable of providing a

References
----------

[1] https://upload.wikimedia.org/wikipedia/commons/4/4d/Xbox-360-Wireless-Controller-White.jpg
[2] http://opengameart.org/content/free-keyboard-and-controllers-prompts-pack