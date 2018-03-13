# **Behavioral Cloning Project** 

## Teaching a car to learn how to drive from your examples  
---

**Behavioral Cloning Project**  
Part of the [Udacity Self Driving Car Nanodegree](http://www.udacity.com/drive)  
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[centerdriving]: ./images/centerdriving.png "Center Driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[modelarch]: ./images/modelarch.png "Model Architecture"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-v13.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model-v13.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I tried various different architectures and went with what worked the best. The NVIDIA architecture worked the best for me.  
My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 235-249) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 237). The image is also cropped using a Keras cropping 2D layer (code line 238).  

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 245). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 333). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 306).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I augmented the data with flipped images, and shifted images.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach


My first step was to use a convolution neural network model similar to the NVIDIA architecture I thought this model might be appropriate because the lessons suggested it. It seemed to work well for me. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included dropout layers to combat overfitting. I also trained it for a small number of epochs (=2). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I decided to augment the data. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 235-248) was a slighlty modified version of the NVIDIA model used in the lessons. It consisted of a convolution neural network with the following layers and layer sizes 
* Normalization layer
* Cropping layer  
* Convolutional layer with kernel size 5x5 and depth 24 plus ELU activation layer
* Convolutional layer with kernel size 5x5 and depth 36 plus ELU activation layer
* Convolutional layer with kernel size 5x5 and depth 48 plus ELU activation layer
* Convolutional layer with kernel size 3x3 and depth 64 plus ELU activation layer
* A dense layer of size 120
* A dropout layer with ```keep_prob = 0.5```
* A dense layer of size 80
* A dense layer of size 40
* An output dense layer with 1 neuron

The loss function used was mean square error.  
Here is a visualization of the NVIDIA architecture

![alt text][modelarch]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][centerdriving]

Along with this data I used the dataset provided by Udacity to train the model. For recovery I decided to make use of the left and right camera images. 

<!--I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from situations where it was near the end of the road. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]-->

To augment the data set to reduce the bias present, i.e. large number of samples with steering angle close to 0, I did two things:
1. I flipped images and angles reducing the leftward skew of the data. 
2. I augmented the data by performing random translation shifts. 

I then used a ```generator``` function (lines 102-164) to reduce the bias in the data. It does three things:  
1. one of the three (center, left, right) images is chosen randomly
2. 50% of the time this image is also flipped
3. 50% of the time this image is also shifted using ```translate_shift```  

For the flipped images a steering correction of ```0.25``` was used
The way this generator function is defined allows me to generate a large number of data points. 

I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used a ```batch_size=32``` and a training dataset size of 40,000 for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by overfitting that occured for a higher number of epochs. I used an adam optimizer so that manually setting the learning rate wasn't necessary.

The model works quite well on track 1 and can be seen in the below video. The video generated by ```video.py``` is stored in ```model-v13.mp4```

[![Model driving around track](https://img.youtube.com/vi/2eru35v02uE/0.jpg)](https://www.youtube.com/watch?v=2eru35v02uE)

