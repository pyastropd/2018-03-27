# Machine learning for images and more: Keras and Tensorflow

by *Mario Pasquato*

I will present Keras. Keras is an easier way to use TensorFlow.

TensorFlow is used to build and train neural networks. Keras is more user friendly.

Basically we throw a numpy array with data in, and get a numpy array with predictions out.

-----

## You will need to install tensorflow and keras
This should work:
```bash
sudo pip install tensorflow
sudo pip install keras
```
if not, have a look at [the installation instructions over here](https://keras.io).


## Why Keras?
* You want to build deep learning models (fancy name for neural networks with hidden layers)
* You use Python (2.7 and later versions will work)
* You don't want to learn [TensorFlow](https://www.tensorflow.org)
* You still want to use TensorFlow through Keras

## Scripts in the repo
* `artificial_data.py` Fully connected neural net, classifies artificially generated data
* `artificial_data_regression.py` Fully connected nn, regression
* `cifar10.py` Convolutional nn recognizes images and saves the model
* `cifar10_loadandpredict.py` Loads model and shows images, predictions...
* other scripts

## Data download
Whe you run `cifar10.py` it will download the [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) data. 



