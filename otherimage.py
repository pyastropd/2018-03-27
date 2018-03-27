from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

model = load_model('model_good.hd5')

img = mpimg.imread('testdog.png')

print img.shape

plt.imshow(img)
plt.show()

#this gives an error... figure out why
p = model.predict(img)