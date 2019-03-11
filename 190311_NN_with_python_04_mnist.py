# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:07:52 2019

@author: ktm
"""

#%% 다층 퍼셉트론


#%%
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
print(time.time())

#%%
''' %%time  in jupyter'''
def nowtime(past):
    now = time.time()
    print(now)
    print('period[second] : {}'.format(now - past))
    return now

#%%
pasttime=0

#%%
pasttime = nowtime(pasttime)

#%% Keras MNIST
from keras.datasets import mnist


#%%
((X_train, y_train), (X_test, y_test)) = mnist.load_data()

#%%
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#%%
print('label = {0}'.format(y_train[0:15])) 

#%%
figure, axes = plt.subplots(nrows=3, ncols=5)
figure.set_size_inches(18,12)

cnt = 0
row = 0
for i in range(3):
    axes[i][0].matshow(X_train[0])
    axes[i][1].matshow(X_train[1])
    axes[i][2].matshow(X_train[2])
    axes[i][3].matshow(X_train[3])
    axes[i][4].matshow(X_train[4])
    

#%%
figure, axes = plt.subplots(nrows=3, ncols=5)
figure.set_size_inches(18,12)

cnt = 0
row = 0
for i in range(3):
    axes[i][0].matshow(X_train[cnt+0])
    axes[i][1].matshow(X_train[cnt+1])
    axes[i][2].matshow(X_train[cnt+2])
    axes[i][3].matshow(X_train[cnt+3])
    axes[i][4].matshow(X_train[cnt+4])
    cnt+=5
    
#%% 전처리
X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)
print(X_train.shape, X_test.shape)

#%%
y_train_hot = np.eye(10)[y_train]
print(y_train[0:5])
y_train_hot[0:5]

#%%
y_test_hot = np.eye(10)[y_test]
print(y_test[0:5])
y_test_hot[0:5]

#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%%
x_value = np.linspace(start =-10, stop = 10)
x_value
y_value = sigmoid(x_value)
plt.plot(x_value, y_value)
#%%
num_epoch =10
learning_rate = 0.0001

w1 = np.random.uniform(low=-1, high=1, size=(28*28, 1000))
w2 = np.random.uniform(low=-1, high=1, size=(1000, 10))

#%%
for epoch in range(num_epoch):
    
    # forward propagation
    layer1 = X_train.dot(w1)
    layer1_out = sigmoid(layer1)
    layer2 = layer1_out.dot(w2)
    layer2_out = sigmoid(layer2)
    
    #print(layer2_out.shape)
    
    predict = np.argmax(layer2_out, axis=1)
    error = (predict != y_train).mean()
    
    if error < 0.01:
        break
    
    if epoch % 2 ==0:
        print('epoch = {0:3}, error = {1:.5f}'.format(epoch, error))
        
    # Back propagation
    d2 = layer2_out - y_train_hot
    d1 = d2.dot(w2.T)* layer1_out * (1-layer1_out)
    
    w2 = w2 - learning_rate * layer1_out.T.dot(d2)
    w1 = w1 - learning_rate * X_train.T.dot(d1)
    
print('epoch = {0:3}, error = {1:.5f}'.format(epoch, error))


























































    