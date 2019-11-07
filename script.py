# importing the dataset
from keras.datasets import cifar10

#loading the data in train and test spliting

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#now lets explore the data

# now let us find the shape of the following data exploring train test

print('X_train shape', X_train.shape)

# we get (50000,32,32,3) which means 5000 images with 32 height, 32 depth and 3 depth(rgb)


#print('y_train shape', y_train.shape) # (50000,1)

#print('X_test shape', X_test.shape)  #(10000,32,32,3)

#print('y_test shape', y_test.shape) #(10000,1)

#print(X_train[0])y

import matplotlib.pyplot as plt

img = plt.imshow(X_train[1])

#print(img)

print('the label is', y_train[1]) # since the 9th one is the truck hence the image is a truck

# Now we do the conversion to neglect the over fitting
# we perform the one hot encoding
import keras
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10) # 10 defines the number of functiond to the clsss it is used to define the number of parameters

print('the one hot label is', y_train_one_hot[1]) # proceeing our label

# processing our image

#we simply need todivide our 255 pixel
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255
X_test = X_test/255

# we have done the two steps change the label to one hot encoding and scale the image picture between 0 and 1

# BUILDING AND TRAINING THE CONVOLUTION NEURAL NETWORK

from keras import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D

model = Sequential()

model.add(Conv2D(32,(3,3), activation = 'relu' ,padding = 'same', input_shape = (32,32,3)))


model.add(Conv2D(32,(3,3), activation = 'relu' ,padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

# the four layers are added now we add the next four layers

model.add(Conv2D(64, (3,3), activation = 'relu',padding = 'same'))

model.add(Conv2D(64, (3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation  = 'softmax'))

model.summary()

#compiling the model

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

hist = model.fit(X_train, y_train_one_hot, batch_size = 32, epochs = 20, validation_split = 20)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

model.evaluate(x_test, y_test_one_hot)[1]

model.save('my_cifar10_model.h5')

from keras.models import load_model
model = load_model('my_cifar10_model.h5')



my_image = plt.imread("cat.jpg")

#The first thing we have to do is to resize the image of our cat so that we can fit it into our model (input size of 32 * 32 * 3). Instead of coding a resize function ourselves, let’s download a package called ‘scikit-image’ which would help us with that function.


from skimage.transform import resize

my_image_resized = resize(my_image, (32,32,3))

#We can visualize our resized image like this:

img = plt.imshow(my_image_resized)


#Note that the resized image has pixel values already scaled between 0 and 1, so we need not apply the pre-processing steps that we previously did for our training image. And now, we see what our trained model will output when given an image of our cat, using the code of model.predict:

import numpy as np
probabilities = model.predict(np.array( [my_image_resized,] ))

#This might look confusing, but model.predict expects a 4-D array instead of a 3-D array (with the missing dimension being the number of training examples). This is consistent with the training set and test set that we had previously. Thus, the np.array(...) code is there to change our current array of my_image_resized into a 4-D array before applying the model.predict function.

#The outputs of the code above are the 10 output neurons corresponding to a probability distribution over the classes. If we run the cell

probabilities

#we should be able to see the probability predictions for all the classes:
#Probability prediction for our cat image


number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])