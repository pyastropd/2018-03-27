from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

train_x = np.random.rand(1000, 2) #2 variables, 1000 data rows, uniform random in [0,1]
train_y = train_x[:,1]*(1.0 - train_x[:,0]) #y = x_1 * (1 - x_0)

#Set up the neural net
nn = Sequential() #basic model, stacked layers that feed into each other
nn.add(Dense(10, input_dim=2, activation = 'tanh')) #2 inputs into 10 neurons of the first layer
nn.add(Dense(10, activation = 'tanh')) #one more fully connected layers, 10 neurons
nn.add(Dense(1, activation = 'tanh')) #one neuron to output

nn.summary() #tells us how many degrees of freedom we have, and the general architecture

#Set up the training parameters
nn.compile(loss='mean_squared_error', optimizer='sgd') #minimize mean squared error via stochastic gradient descent

#Train the net (sees the data 100 times)
nn.fit(train_x, train_y, epochs=100, batch_size=10)

test_x = np.random.rand(1000, 2)
test_y = test_x[:,1]*(1.0 - test_x[:,0])

pred_y = nn.predict(test_x)
pred_y.shape = [pred_y.size,]

plt.scatter(test_y, pred_y)
plt.show()
