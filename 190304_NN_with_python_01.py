# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 09:06:03 2019

@author: ktm
"""

#%%
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

pd.set_option('display.float_format', '{:.2f}'.format)

#%% 01. 데이터 불러오기
boston = load_boston() # 딕셔네리
print(type(boston))
print(boston.items())
print(boston.keys()) # 딕셔너리임을 확인 가능
print(boston.feature_names) # 컬럼명
print(boston.data.shape)
print(boston.target) # 타겟값

x = boston.data
#x=boston['data']
y = boston.target

#%% 02.
data = pd.DataFrame(x, columns=boston.feature_names)
data.head()
data['Price'] = y
data.shape

#%% 03. 그레디언트 디센트
num_epoch = 10000
learning_rate = 0.000005

x1 = data['CRIM'].values
x2 = data['ZN'].values

w1 = np.random.uniform(low=0.0, high=1.0)
w2 = np.random.uniform(low=0.0, high=1.0)
b = np.random.uniform(low=0.0, high=1.0)

#%% 04.
# y = wx + b
# y = w1x1 + w2x2 + b 구현
for epoch in range(num_epoch):
    y_predict = x1 * w1 + \
                x2 * w2 + b
    # loss function            
    err = np.abs(y_predict - y).mean() # : cost,  506행의 데이터가 들어가기 때문에 mean 취함
    # 멈추는 시점 정의
    if err < 3:
        break
    
    w1 = w1 - learning_rate * ((y_predict - y)*x1).mean()  # err를 w1으로 미분한 것을 뺀다.
    w2 = w2 - learning_rate * ((y_predict - y)*x2).mean()
    b = b - learning_rate*((y_predict - y).mean())
    
    if epoch % 1000 == 0:
        #print('epoch:', epoch, 'err:', err, \
        #      'w1:', w1, 'w2:', w2, 'b:', b)
        print('epoch:{0:.2f} \n err: {1:.2f} \n w1:{2:.2f} \n w2:{3:.2f} \n b:{4:.2f}'.format(epoch, err , w1, w2, b))
             
        
print('w1: ', w1,'w2: ', w2,'err: ', err)

#%%


#%%