# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

#%% 데이터 준비
x1 = np.random.uniform(low=-1.0, high=1.0, size=100)
x2 = np.random.uniform(low=-1.0, high=1.0, size=100)
b = np.random.uniform(low=-1.0, high=1.0)*0.2

print(x1.shape, x2.shape, b)

#%% 02. y를 준비
y = x1*0.3 +x2*0.5 +0.1+b
y = y >=0.0
y
#%%
plt.scatter(x1, x2, c=y)

#%% 시그모이드 함수 만들기
def sigmoid(n):
    return 1/(1+np.exp(-n))

#%%
num_epoch = 1000

w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)
b = np.random.uniform(low=0.0, high=1.0)

#%%


#%%
err_sum = 0
total = 0
for epoch in range(num_epoch):
    y_predict = sigmoid(x1*w1 + x2*w2 +b)
    
    predict = (y_predict >= 0.5)
    actual = y
    
    # loss function
    err = (predict != actual).mean()
    
    # err 갯수 확인 추가
    for i in range(100):
        total += 1
        if predict[i] != actual[i] :
            err_sum += 1
    
    # 멈추는 시점 정의
    if err < 0.01:
        break
    
    if epoch % 100 == 0:
        print('{0} error:{1:.3f} w1:{2:.3f} w2:{3:.3f} '.format(epoch, err, w1, w2) )
    
    w1 = w1 - ((y_predict - y)*x1).mean()
    w2 = w2 - ((y_predict - y)*x2).mean()
    b = b - (y_predict - y).mean()

print('err_sum:', err_sum)
print('total:', total)
print('{0} {1} {2} {3}'.format(epoch, err, w1, w2))

# 결과 err 0.29

#%%

import pandas as pd

y_predict = x1*w1 + x2*w2 + b
y_predict = sigmoid(y_predict)

predict = (y_predict >=0.5)

res = {'x1':x1, 'x2':x2, 'y_actual':y, 'y_predict':predict}

data = pd.DataFrame(res)
data

#%% 러닝 레이트 넣어서 다시 하
err_sum = 0
total = 0
for epoch in range(num_epoch):
    y_predict = sigmoid(x1*w1 + x2*w2 +b)
    
    predict = (y_predict >= 0.5)
    actual = y
    
    # loss function
    err = (predict != actual).mean()
    
    # err 갯수 확인 추가
    for i in range(100):
        total += 1
        if predict[i] != actual[i] :
            err_sum += 1
    
    # 멈추는 시점 정의
    if err < 0.01:
        break
    
    if epoch % 100 == 0:
        print('{0} error:{1:.3f} w1:{2:.3f} w2:{3:.3f} '.format(epoch, err, w1, w2) )
    
    w1 = w1 - 0.01*((y_predict - y)*x1).mean()
    w2 = w2 - 0.01*((y_predict - y)*x2).mean()
    b = b - (y_predict - y).mean()

print('err_sum:', err_sum)
print('total:', total)
print('{0} {1} {2} {3}'.format(epoch, err, w1, w2))