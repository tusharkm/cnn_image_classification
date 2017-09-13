#part-1 - Building CNN


from keras.models import Sequential
from keras.layers import Convolution2D    # 2D for images 3D for Video
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initializing the CNN
# Step-1 - Convolution
from numpy.core.tests.test_numeric import test_outer_out_param

classifier =Sequential()

#Step-1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation="relu"));


#step-2 Pooling
classifier.add(MaxPooling2D(2,2))


#Part-3 Improving the model:  by adding convolution layer
classifier.add(Convolution2D(32,3,3,activation="relu"));
classifier.add(MaxPooling2D(2,2));


#Step-3 Flatten
classifier.add(Flatten())

#Step-4 Full Connection- creating ANN
#hidden layer out put layer
classifier.add(Dense(output_dim =128,activation="relu"))

#adding output layer
classifier.add(Dense(output_dim =1,activation="sigmoid"))


#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Part2: Fitting the CNN to the images


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,  # all  pixel value are   made in the range 1-255
        shear_range=0.2, #applied random transfectiom
        zoom_range=0.2,# applied random zooom
        horizontal_flip=True) # images are flipped horizontally


test_datagen = ImageDataGenerator(rescale=1./255)

#create traing set
training_set = train_datagen.flow_from_directory('C:/Users/Tushu/Google Drive/Programs/Python_Programs/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size=(64,64), 
                                                 batch_size=32, 
                                                 class_mode='binary')
#create validation set
test_set = test_datagen.flow_from_directory('C:/Users/Tushu/Google Drive/Programs/Python_Programs/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
        steps_per_epoch=8000, # number of images in traing set we have 80000 images
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) # number of images in test set

#================================================================================
#part4 : Making new prediction

import numpy as np
from keras.preprocessing import image
#load the image
test_image=image.load_img('C:/Users/Tushu/Google Drive/Programs/Python_Programs/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64, 64))



test_image=image.img_to_array(test_image)

test_image=np.expand_dims(test_image,axis=0)

result=classifier.predict(test_image)

training_set.class_indices 


if result[0][0]== 1:
    prediction= 'DOG'
else :
    prediction='CAT'

