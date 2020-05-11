# -*- coding: utf-8 -*-

# K nearest neighbours classification of hand-drawn digits

from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from pathlib import Path
import os, sys

# Load data

print(os.getcwd())
dataFolder = Path(".././data/")
data = MNIST(dataFolder)
images, labels = data.load_training()

learning = KNeighborsClassifier()

# Train the model
trainX = images[:10000]
trainY = labels[:10000]
learning.fit(trainX, trainY)

# Test on the next 1000 images
testX = images[10000:10100]
expected = labels[10000:10100].tolist()
predicted = learning.predict(testX)

titles_options = [("Confusion matrix", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(predicted, testX, expected,
                                 #display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()