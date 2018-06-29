import csv
import numpy as np
import cv2
import sklearn


lines = []
# read csv file
with open("./collectData/driving_log.csv") as csvfile:
    spamreader = csv.reader(csvfile)
    for line in spamreader:
        lines.append(line)

images = []
measurements = []
for line in lines[0:101]:
    source_path = line[0]
    file_name = source_path.split('\\')[-1]
    current_path = "./collectData/IMG/"+file_name
    image = cv2.imread(current_path)    
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # data augmentation: flip images
    image_flip = cv2.flip(image, flipCode=0)
    measurement_flip = -measurement
    images.append(image_flip)
    measurements.append(measurement_flip)

    # data augmentation; including left and right camera images
    # left
    source_path = line[1]
    file_name = source_path.split('\\')[-1]
    current_path = "./collectData/IMG/"+file_name
    image_left = cv2.imread(current_path)
    measurement_left = measurement+0.2
    images.append(image_left)
    measurements.append(measurement_left)
    # right (similarly)
    source_path = line[2]
    file_name = source_path.split('\\')[-1]
    current_path = "./collectData/IMG/"+file_name
    image_right = cv2.imread(current_path)
    measurement_right = measurement-0.2
    images.append(image_right)
    measurements.append(measurement_right)
    
    
X_train = np.array(images)
y_train = np.array(measurements)
    

# load images using generators



# define model 
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten, Lambda
from keras.layers.convolutional import Cropping2D, Convolution2D


model = Sequential()    # Input shape 160x320x3
model.add( Cropping2D( cropping=((65,22),(0,0)), input_shape=(160,320,3) ) )# crop images
model.add( Lambda(lambda x: x/127.5-1.0) )  # normalization
model.add(Convolution2D(24,5,5, border_mode='valid',subsample=(2,2))) # CNN 24@31x98
model.add(Convolution2D(36,5,5, border_mode='valid',subsample=(2,2))) # CNN 36@14x47
model.add(Convolution2D(48,5,5, border_mode='valid',subsample=(2,2))) # CNN 48@5x22
model.add(Convolution2D(64,3,3, border_mode='valid',subsample=(1,1))) # CNN 64@3x20
model.add(Convolution2D(64,3,3, border_mode='valid',subsample=(1,1))) # CNN 64@1x18
model.add(Flatten())  # Flatten
model.add(Dense(100)) # Fully Connected 100
model.add(Dense(50))  # Fully Connected 50
model.add(Dense(10)) # Fully Connected 10
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)  # train model

model.save('model.h5')
# model.load_weights('my_model_weights.h5')
# or
# del model
# model = load_model('my_model.h5')


# validate model


"""
# visualization
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data =
    validation_generator,
    nb_val_samples = len(validation_samples),
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""