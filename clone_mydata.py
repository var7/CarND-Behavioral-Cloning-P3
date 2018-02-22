import csv
import cv2
import numpy as np

lines = []

with open('./my_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './my_data/IMG/' + filename
	image = cv2.imread(current_path)
	if image is None:
		print('Incorrect path', current_path)
	else:
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
		image_flipped = np.fliplr(image)
		measurement_flipped = -measurement
		images.append(image)
		measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
# print(X_train[0])
# print(X_train.shape)
# print(y_train.shape)

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
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch=5)

model.save('nvidia_mydatamodel.h5')
exit()
