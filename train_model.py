import os
import csv
FOLDER = './data/'
samples = []
with open(FOLDER+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

IMGPATH = FOLDER + 'IMG/'
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
	    for line in batch_samples:
          source_path = line[0]
          filename = source_path.split('/')[-1]
          current_path = IMGPATH + filename
          left_filename = line[1].split('/')[-1]
          left_path = IMGPATH + left_filename
          right_filename = line[2].split('/')[-1]
          right_path = IMGPATH + right_filename
          image = cv2.imread(current_path)
          left_image = cv2.imread(left_path)
          right_image = cv2.imread(right_path)
          if image is None:
          	print('Incorrect path', current_path)
          else:
          	images.extend([image, left_image, right_image])
          	measurement = float(line[3])
          	steering_left = measurement + correction
          	steering_right = measurement - correction
          	measurements.extend([measurement, steering_left, steering_right])
          	image_flipped = np.fliplr(image)
          	left_image_flipped = np.fliplr(left_image)
          	right_image_flipped = np.fliplr(right_image)
          	measurement_flipped = -measurement
          	steering_left_flipped = -steering_left
          	steering_right_flipped = -steering_right
          	images.extend([image_flipped, left_image_flipped, right_image_flipped])
          	measurements.extend([measurement_flipped, steering_left_flipped, steering_right_flipped])
    # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation = "relu"))
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Dropout(0.25))
model.add(Conv2D(48,5,5,subsample=(2,2), activation='relu'))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Conv2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.50))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')


history_object = model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('2nvidia_model.png', bbox_inches='tight')
model.save('2nvidia_model.h5')
exit()
