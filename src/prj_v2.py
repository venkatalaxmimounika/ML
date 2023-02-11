import numpy as np
import gzip
import os
import struct
import sys
import matplotlib.pyplot as plt

w = np.array(np.random.uniform(-1,1,size=(10,784)))
eta = 0.0001
epsilon = 0.001
epoch = 0
tot_epochs = []
errors = []
#errors.append(0)
count = 0
n = 60000
d_output = []



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


# step activation function definition
def my_activation(local_field): 
    return_this = np.zeros(local_field.shape[0]) 
    dum = np.where(local_field >= 0)[0] 
    return_this[dum] = 1 
    return return_this
 


train_data = read_images('train-images-idx3-ubyte.gz')
train_label = read_labels('train-labels-idx1-ubyte.gz')
test_data = read_images('t10k-images-idx3-ubyte.gz')
test_label = read_labels('t10k-labels-idx1-ubyte.gz')





#calculating induced local fields
while epoch < 10000:
    epoch_error = 0
    count = 0
    for i in range(n):
        xi = train_data[i]
        xi.resize(784,1)
        v = np.array(np.dot(w,xi))
        vj = np.argmax(v)
        if train_label[i] != vj:
            #errors[epoch] = errors[epoch] + 1
            count = count + 1
            epoch_error = epoch_error + 1
    print("epoch", epoch, "  count ", count)       
    errors.append(epoch_error)
    tot_epochs.append(epoch)
    #print("epoch",epoch)
    #print("errors(epoch)",errors[epoch])
    epoch = epoch + 1

    # update weights       
    for i in range(n):
        xi = train_data[i]
        xi.resize(784,1)
        u = np.dot(w,xi)
        v = my_activation(u)
        v.resize(10,1)
        d = np.zeros((1,10))
        di = d.transpose()
        di[train_label[i]] = 1
        d_output = di
        diff = np.subtract(d_output,v)
        xi.resize(1,784)
        w_update = (np.dot(diff,xi))*eta
        w = w + w_update
        
        if errors[epoch - 1]/n > epsilon:
            continue
        


test_errors = 0
tot_test_errors = []

#test on the corresponding test set images and labels
for i in range(10000):
    xi2 = test_data[i]
    xi2.resize(784,1)
    v2 = np.dot(w,xi2)
    vj2 = v2.argmax()
    if vj2 != test_label[i]:
        test_errors = test_errors + 1
    tot_test_errors.append(test_errors)   
#print("test errors",test_errors)
print("test errors using the trained model ",test_errors)
accuracy = ((10000 - test_errors) / 10000) * 100
print("test accuracy ", accuracy)

#graph plotting

plt.xlim((0,100))
plt.ylim((0,60000))
plt.xlabel("Number of epoch")
plt.ylabel("Number of misclassifications")
plt.title("eta=1 , n=50 , epsilon=0.01")
plt.plot(tot_epochs,errors)


plt.show()