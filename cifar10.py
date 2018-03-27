#based on https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
#much smaller to train on laptop; removed dropout and data augmentation
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10

#read the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#set learning parameters
batch_size = 500
num_classes = 10
epochs = 5

#convert to appropriate format (1-hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#set up the net
model = Sequential() #empty sequential model

#1st layer is a convolutional layer with 8 neurons on 3x3 fields
model.add(Conv2D(8, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #followed by max pooling

#2nd layer also a convolutional layer, with max pooling, same as before
model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#3rd layer is fully connected
model.add(Flatten())
model.add(Dense(12))
model.add(Activation('relu'))

#last layer outputs the probability of predicted classes
#(has num_classes neurons, softmax activation)
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#see the model
model.summary()

# initiate RMSprop optimizer, see http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

#set up training
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#scale the pixel values to [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

