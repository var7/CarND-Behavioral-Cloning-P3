# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains files used for the completion of the Behavioral Cloning Project.

In this project, I've used deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

Udacity provided a simulator where you can steer a car around a track for data collection. I use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

Check out the [writeup](https://github.com/var7/CarND-Behavioral-Cloning-P3/blob/master/writeup.md) for this project.

To meet specifications, the project consists of five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model-v13.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This code requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py

## Running the model

### `drive.py`

To run the trained model use it with drive.py using this command:

```sh
python drive.py model-v13.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.


## Results
The model works quite well on track 1 and can be seen in the below video. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/2eru35v02uE" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


