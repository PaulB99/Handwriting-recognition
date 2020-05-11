# -*- coding: utf-8 -*-

# K nearest neighbours classification of hand-drawn digits

from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from pathlib import Path
from sklearn.metrics import accuracy_score

# Load data

dataFolder = Path(".././data/")
data = MNIST(dataFolder)
data.gz = True
images, labels = data.load_training()

learning = KNeighborsClassifier()

# Train the model
trainX = images[:20000]
trainY = labels[:20000]
learning.fit(trainX, trainY)

# Test on the next 1000 images
testX = images[10000:11000]
expected = labels[10000:11000].tolist()
predicted = learning.predict(testX)

print("K Nearest Neighbours accuracy: ", accuracy_score(expected, predicted))