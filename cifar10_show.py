import matplotlib.pyplot as plt
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#show some of the images
for i in range(0, 10):
    plt.imshow(x_train[i,:,:,:])
    plt.show()