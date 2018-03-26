import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

np.random.seed(seed = 37)
train = np.random.rand(10000, 4)
label = np.asarray([round(j*j) for j in train[:,3]]) #np.random.randint(2, size=(512, 1))

print train.shape
print label.shape

neuralNet = Sequential()
neuralNet.add(Dense(8, input_dim=4))
neuralNet.add(Activation('relu'))
neuralNet.add(Dense(8))
neuralNet.add(Activation('relu'))
neuralNet.add(Dense(1))
neuralNet.add(Activation('sigmoid'))

neuralNet.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

neuralNet.fit(train, label, epochs=100, batch_size=32)