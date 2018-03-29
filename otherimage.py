#demonstration of classification of an image outside of the
#cifar10 dataset ('testdog.png')
#the model misclassifies it as a frog, and predicts an higher
#probability for it to be a cat than a dog
#even though this dog was voted as the cutest dog ever
#in a competition hosted by the russian social network VKontatke
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

labeldict = { 0 : 'airplane',
	          1 : 'automobile',
              2 : 'bird',
              3 : 'cat',
              4 : 'deer',
              5 : 'dog',
              6 : 'frog',
              7 : 'horse',
              8 : 'ship',
              9 : 'truck'
}

model = load_model('model_good.hd5')

img = mpimg.imread('testdog.png')
img = img[:,:,0:3]

img.shape = [1,32,32,3] #need to do this to feed it into the model

p = model.predict(img)

predicted_class_y = np.argmax(p)

print p
print predicted_class_y
print labeldict[predicted_class_y] #this dog is a frog
