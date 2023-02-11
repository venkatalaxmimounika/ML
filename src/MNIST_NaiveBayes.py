from scipy.stats import mode
import numpy as np
#from mnist import MNIST

from time import time
import pandas as pd
import os
import matplotlib.pyplot as matplot
import matplotlib
#matplotlib inline

import random
matplot.rcdefaults()
from IPython.display import display, HTML
from itertools import chain
from sklearn.metrics import confusion_matrix
import seaborn as sb

import gzip
import os
import struct
import sys

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data/')



# reading files
def read_images(fl):
    with gzip.open(fl, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
        .reshape((image_count, row_count, column_count))
        return images
        
def read_labels(fl):
    with gzip.open(fl, 'r') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels



train_data = read_images('train-images-idx3-ubyte.gz')
train_label = read_labels('train-labels-idx1-ubyte.gz')
test_data = read_images('t10k-images-idx3-ubyte.gz')
test_label = read_labels('t10k-labels-idx1-ubyte.gz')


train = train_data
trlab = train_label
test = test_data
tslab = test_label


#validation = mnist.validation.images
#test = mnist.test.images

#trlab = mnist.train.labels
#vallab = mnist.validation.labels
#tslab = mnist.test.labels

#train = np.concatenate((train, validation), axis=0)
#trlab = np.concatenate((trlab, vallab), axis=0)

train = train * 255
test = test * 255

def naivebayes(train, train_lb, test, test_lb, smoothing):
    n_class = np.unique(train_lb)
    tr = train
    te = test
    tr_lb = train_lb
    te_lb = test_lb
    smoothing = smoothing
    st = time()
    m, s, prior, count = [], [], [], []
    for i, val in enumerate(n_class):
        sep = [tr_lb == val]
        print(sep)
        count.append(len(tr_lb[sep]))
        prior.append(len(tr_lb[sep]) / len(tr_lb))
        m.append(np.mean(tr[sep], axis=0))
        s.append(np.std(tr[sep], axis=0))


    pred = []
    likelihood = []
    #prtab = []
    lcs = []
	
    for n in range(len(te_lb)):
        classifier = []
        sample = te[n] #test sample
        ll = []
            for i, val in enumerate(n_class):
                m1 = m[i]
                var = np.square(s[i]) + smoothing
                prob = 1 / np.sqrt(2 * np.pi * var) * np.exp(-np.square(sample - m1)/(2 * var))
                #prtab.append(prob)
                result = np.sum(np.log(prob))
                classifier.append(result)
                ll.append(prob)
            
        pred.append(np.argmax(classifier))
        likelihood.append(ll)
        lcs.append(classifier)

    return pred, likelihood

	
def error_rate(confusion_matrix):
    a = confusion_matrix
    b = a.sum(axis=1)
    df = []
    for i in range(0,10):
        temp = 1-a[i][i]/b[i]
        df.append(temp)
    
    df = pd.DataFrame(df)
    df.columns = ['% Error rate']
    return df*100
	
	
nb = naivebayes(train=train, train_lb=trlab, test=test, test_lb=tslab, smoothing=1000)
nb_pred = nb[0]
cm = confusion_matrix(tslab, nb_pred)
print("Test Accuracy:", round((sum(np.diagonal(cm)) / len(nb_pred)) * 100, 4), '%')



#cm # X-axis Predicted vs Y-axis Actual Values
matplot.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
matplot.xlabel("Predicted")
matplot.ylabel("Actual")
matplot.title("Confusion Matrix")
matplot.show()

error_rate(cm)


likeli = nb[1]
likli = likeli[9999]
matplot.subplots(2,5, figsize=(24,10))
for i in range(10):
    l1 = matplot.subplot(2, 5, i + 1)
    l1.imshow(likli[i].reshape(28, 28), interpolation='nearest',cmap=matplot.cm.RdBu)
    l1.set_xticks(())
    l1.set_yticks(())
    l1.set_xlabel('Class %i' % i)
matplot.suptitle('Conditional probability images for each class (according to Likelihood): Rightly Classified as 6')
matplot.show()


likli = likeli[9998]
matplot.subplots(2,5, figsize=(24,10))
for i in range(10):
    l1 = matplot.subplot(2, 5, i + 1)
    l1.imshow(likli[i].reshape(28, 28), interpolation='nearest',cmap=matplot.cm.RdBu)
    l1.set_xticks(())
    l1.set_yticks(())
    l1.set_xlabel('Class %i' % i)
matplot.suptitle('Conditional prob images for each class (according to Likelihood): Wrongly Classified as 8, when it is 5')
matplot.show()
		
	