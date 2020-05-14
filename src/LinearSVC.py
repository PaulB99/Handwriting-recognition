# -*- coding: utf-8 -*-

# Linear support vector classification of hand written digits

from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import LinearSVC
from pathlib import Path
from sklearn.metrics import accuracy_score
import seaborn as sn
import pandas as pd

# Load data
dataFolder = Path(".././data/")
data = MNIST(dataFolder)
data.gz = True
images, labels = data.load_training()

learning = LinearSVC()

# Train the model
trainX = images[:20000]
trainY = labels[:20000]
learning.fit(trainX, trainY)

# Test on the next 1000 images
testX = images[20000:22000]
expected = labels[20000:22000].tolist()
predicted = learning.predict(testX)

# Print the accuracy score
print("K Nearest Neighbours accuracy: ", accuracy_score(expected, predicted))

confusion=metrics.confusion_matrix(expected,predicted)
print(confusion)
plt.matshow(confusion, cmap='binary')

# Seaborn heatmap as a confusion matrix
df = pd.DataFrame(confusion, range(10), range(10))
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
confFig = sn.heatmap(df, annot=True, fmt="d", annot_kws={"size": 16})

# Save figure to file
fig = confFig.get_figure()
fig.savefig("LinearSVC.png")
