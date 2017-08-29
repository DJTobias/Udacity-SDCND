import csv
import cv2
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense,Lambda, Dropout, ELU,Activation
from keras.layers import Convolution2D,Cropping2D,MaxPooling2D,Conv2D 

path_2_images = 'data/IMG/'

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                steering_center = float(batch_sample[3])
                #create additional steering images for center camera by using left and right
                #camera images with an offset to imitate additional turning data
                steering_left = steering_center + 0.2
                steering_right = steering_center - 0.2

                _img_center = cv2.imread(path_2_images + batch_sample[0].split('/')[-1])
                image_center =cv2.cvtColor(_img_center,cv2.COLOR_BGR2RGB)

                _img_left = cv2.imread(path_2_images + batch_sample[1].split('/')[-1])
                image_left =cv2.cvtColor(_img_left,cv2.COLOR_BGR2RGB)
                
                _img_right = cv2.imread(path_2_images + batch_sample[2].split('/')[-1])
                image_right =cv2.cvtColor(_img_right,cv2.COLOR_BGR2RGB)
          
                images.extend([image_center, image_left, image_right])
                angles.extend([steering_center, steering_left, steering_right])
                
                images.extend([np.fliplr(image_center), np.fliplr(image_left), np.fliplr(image_right)])
                angles.extend([-steering_center, -steering_left, -steering_right])
                              
            X_train = np.array(images)
            y_train = np.array(angles)
                       
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Nvidia End to End with reduced fully connected layer
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))#remove top 70 pixels and bottem 25 pixel row
model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2) ))
model.add(Conv2D(48, 3, 3, activation='relu', subsample=(2, 2) ))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(12))
model.add(Dense(1))


b_size = 32
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse',optimizer=adam)
model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples)//b_size, 
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples)//b_size,
                    nb_epoch=35)


model.save('model.h5')
