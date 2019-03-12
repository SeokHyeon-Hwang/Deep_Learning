# -*- coding: utf-8 -*-

#%%
import numpy as np
import matplotlib.pyplot as plt
import time

#%%
def nowtime(past):
    now = time.time()
    print(now)
    print('period[second] : {}'.format(now - past))
    return now

#%%
pasttime = nowtime(pasttime)

#%%
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
#%%
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
pasttime = 0
pasttime = nowtime(pasttime)
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
pasttime = nowtime(pasttime)

#%%
import pandas as pd


#%%
now = time.time()
print(now)
#%%
time.localtime(time.time())
#%%
time.strftime('%Y-%m-%d', time.localtime(time.time()))
#%%
time.strftime('%c', time.localtime(time.time()))
#%%
import datetime
datetime.datetime.today()
#%%
#시작부분 코드
import time
start_time = time.time() 

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

#종료부분 코드
print("start_time", start_time) #출력해보면, 시간형식이 사람이 읽기 힘든 일련번호형식입니다.
print("--- %s seconds ---" %(time.time() - start_time))

#%%
#1
import timeit
start = timeit.default_timer()
 
# 실행 코드
 
stop = timeit.default_timer()
print(stop - start)

#2
import time
startTime = time.time()
 
# 실행 코드
 
endTime = time.time() - startTime
print(endTime)

#3
import time

start_vect=time.time()
print("training Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

#%%
import pandas as pd

l1 = X_test.dot(w1)
o1 = sigmoid(l1)
l2 = o1.dot(w2)
o2 = sigmoid(l2)
#%%
figure, axes = plt.subplots(nrows=3, ncols=5)
figure.set_size_inches(18, 12)

plt.gray()
print("label = {0}".format(y_test[0:15]))

cnt = 0
row = 0
for i in range(3):
    axes[i][0].matshow(X_test[cnt + 0].reshape(28, 28))
    axes[i][1].matshow(X_test[cnt + 1].reshape(28, 28))
    axes[i][2].matshow(X_test[cnt + 2].reshape(28, 28))
    axes[i][3].matshow(X_test[cnt + 3].reshape(28, 28))
    axes[i][4].matshow(X_test[cnt + 4].reshape(28, 28))
    cnt += 5

predict = np.argmax(o2, axis=1)

accuracy = (y_test == predict).mean()
print("Accuracy = {0:.5f}".format(accuracy))

pd.DataFrame({'y(actual)': y_test, 'y(predict)': predict}).head(15)

#%%
from datetime import datetime


#%%
past_time=0

#%%
past_time = nowtime(past_time)

learning_rate = 0.00001   # 0.00001

w1 = np.random.uniform(low=-0.058, high=+0.058, size=(784, 1000))
w2 = np.random.uniform(low=-0.077, high=+0.077, size=(1000, 10))

b1 = np.random.uniform(low=0, high=0, size=(1, 1000))
b2 = np.random.uniform(low=0, high=0, size=(1, 10))

num_epoch = 100

for epoch in range(num_epoch):
    # Forward propagation
    l1 = X_train.dot(w1) + b1
    o1 = sigmoid(l1)
    l2 = o1.dot(w2) + b2
    o2 = sigmoid(l2)
    
    predict = np.argmax(o2, axis=1)
    error = (predict != y_train).mean()
    if error < 0.1:
        break

    if epoch % 10 == 0:
        print("{0:3} error = {1:.5f}".format(epoch, error))
        print("{0:3} w1(mean) = {1:.5f}, w1(std) = {2:.5f}".format(epoch, w1.mean(), w1.std()))
        print("{0:3} w2(mean) = {1:.5f}, w2(std) = {2:.5f}".format(epoch, w2.mean(), w2.std()))
        print("{0:3} b1(mean) = {1:.5f}, b1(std) = {2:.5f}".format(epoch, b1.mean(), b1.std()))
        print("{0:3} b2(mean) = {1:.5f}, b2(std) = {2:.5f}".format(epoch, b2.mean(), b2.std()))
        print("----" * 11)

    # Backpropagation
    d2 = o2 - y_train_hot
    d1 = d2.dot(w2.T) * o1 * (1 - o1)

    w2 = w2 - learning_rate * o1.T.dot(d2)
    b2 = b2 - learning_rate * d2.mean(axis=0)
    w1 = w1 - learning_rate * X_train.T.dot(d1)
    b1 = b1 - learning_rate * d1.mean(axis=0)

print("{0:3} error = {1:.5f}".format(epoch, error))

past_time = nowtime(past_time)

#%%
import pandas as pd

l1 = X_test.dot(w1)
o1 = sigmoid(l1)
l2 = a1.dot(w2)
o2 = sigmoid(l2)

figure, axes = plt.subplots(nrows=3, ncols=5)
figure.set_size_inches(18, 12)

plt.gray()
print("label = {0}".format(y_test[0:15]))

cnt = 0
row = 0
for i in range(3):
    axes[i][0].matshow(X_test[cnt + 0].reshape(28, 28))
    axes[i][1].matshow(X_test[cnt + 1].reshape(28, 28))
    axes[i][2].matshow(X_test[cnt + 2].reshape(28, 28))
    axes[i][3].matshow(X_test[cnt + 3].reshape(28, 28))
    axes[i][4].matshow(X_test[cnt + 4].reshape(28, 28))
    cnt += 5

predict = np.argmax(o2, axis=1)

accuracy = (y_test == predict).mean()
print("Accuracy = {0:.5f}".format(accuracy))

pd.DataFrame({'y(actual)': y_test, 'y(predict)': predict}).head(15)

#%%

