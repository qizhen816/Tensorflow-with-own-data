# coding: utf-8

# Imports
import gzip # gzip is only needed if the file is compressed
import matplotlib.pyplot as plt
import pickle

# Un-pickle de data
with gzip.open('dataset.pkl.gz', 'rb') as f:
    train_set, test_set = pickle.load(f)

train = train_set
test = test_set

# Example to plot some images
plt.imshow(train[0].reshape((28, 28)), cmap='gray')
plt.show()