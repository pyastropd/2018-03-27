from keras.models import load_model
from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

show_only_n = 10 #how many images to show

#load the saved model from file
model = load_model('model_good.hd5')
model.summary()

#read the data (we will use only test data)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#predict (returns a class probability for each class)
pred_y = model.predict(x_test)

#convert to a class prediction
predicted_class_y = np.argmax(pred_y, axis=1)

#compare real and predicted classes
comp = np.column_stack((y_test, predicted_class_y))

diff = comp[:,1] - comp[:,0]

print 'Misclassifications: ' + str(np.count_nonzero(diff))
print 'Total number of images: ' + str(diff.size)

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

print "Showing results on the first " + str(show_only_n) + " images"
print "Truth, Prediction"
print np.column_stack(([labeldict[index] for index in comp[0:show_only_n,0]], [labeldict[index] for index in comp[0:show_only_n,1]]))

for i in range(0, show_only_n):
    plt.imshow(x_test[i,:,:,:])
    plt.show()
