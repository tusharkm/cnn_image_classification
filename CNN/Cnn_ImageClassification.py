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
#feature detector=32-> no of feature dectector, 3-> row of feature detector, 3-> column of feature detect
#Input_shape=fix the size of all the input images, this fixing is done later we will first create the model
#activalion-> to make the images non linear(image processing is done line 44 onwards)

#step-2 Pooling
classifier.add(MaxPooling2D(2,2))


#Part-3 Improving the model:  by adding convolution layer
classifier.add(Convolution2D(32,3,3,activation="relu"));
classifier.add(MaxPooling2D(2,2));
# input_shape is not specified as the model knows the shpe from previous model
# we can increase the feture detector from 32 to 64 to improve the model prediction further

#Step-3 Flatten
classifier.add(Flatten())

#Step-4 Full Connection- creating ANN
#hidden layer out put layer
classifier.add(Dense(output_dim =128,activation="relu"))
#output_dim is selected by experiecne it is usually a value between input and output node, since we have large number of input we select 128

#adding out put layer
classifier.add(Dense(output_dim =1,activation="sigmoid"))
#activation function: since we have binary op with probability we use sigmoid activation
#and if we add multiple output we would use softmax activation

#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',matrics=['accuarcy'])
#optimizer=stochastic gradient decent algorithm -> adam
#loss='binary_crossentropy': since its a binary classification problem , we use logarithmic function
# if we had more than 2 op we need to use categorical cross function

#Part2: Fitting the CNN to the images
#Using keras documentation, for image augmentation it consisnt of preprocessing tthe image to prevent over fitting.
# if we donot do this we will overfit the model(https://keras.io/)
# To avoid over fitting we need large amout of data, we have around 10000 images that is not enough data o get the performance
# We can do is get more data orwe can use a trick i.e data augmnentation  what it will do is
# it creates many batches of the images on each batch it will apply random tarnsformation images(like rotating, shifting , fliping them )
#hecne we get deverse of this images and hence lot of images to train.
#since it is random transformation our model will not get same images
#hence we get good performance

#code used from keras https://keras.io/preprocessing/image/
# using flow from dirctory , because our structure is identified using files cat and dog folder

from keras.preprocessing.image import ImageDataGenerator
# data transformation for data augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,  # all  pixel value are   made in the range 1-255
        shear_range=0.2, #applied random transfectiom
        zoom_range=0.2,# applied random zooom
        horizontal_flip=True) # images are flipped horizontally


test_datagen = ImageDataGenerator(rescale=1./255)

#create traing set
training_set = train_datagen.flow_from_directory('C:/Users/Tushu/Google Drive/Programs/Python_Programs/Convolutional_Neural_Networks/Convolutional_Neural_Networks/dataset/training_set',
                                                 target_size=(64,64), #here we set the sie of input file images as 64*64
                                                 batch_size=32, #size of the batches the number of cnn goes through CNN after which weight will be updated
                                                 class_mode='binary')#if dpended variable is binary i.e. cats and dog
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
#target size same as input images


test_image=image.img_to_array(test_image)
#coverting image fro 2d to 3d as we did during part 1,input_shape=(64,64,3), i.e converting input img from (64,64) to (64,64,3)


#classifier.predict(test_image)
# we receive an error we need 4 dimension, it is an error of predict function.
# the 4th dimenion is the batch.bcoz the predict function cannot expect single input by itself it expects input in a batch
# modified test image

test_image=np.expand_dims(test_image,axis=0)
#axis index of the new dimension is eqoal to index is zero,
# old dimension = (64,64,3), new dimension=(1,64,64,3)
result=classifier.predict(test_image)
# op in 1

# to check if one is cat or dog
training_set.class_indices #cats=0 dog =1

#simplifying result
if result[0][0]== 1:
    prediction= 'DOG'
else :
    prediction='CAT'

