# -*- coding: utf-8 -*-

#%%
from keras.datasets import mnist

((X_train, y_train), (X_test, y_test)) = mnist.load_data()


#%%
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=2, ncols=5)
figure.set_size_inches(18, 12)

plt.gray()

print('label = {0}'.format(y_train[0:10]))

# 1행 데이터
axes[0][0].matshow(X_train[10])
axes[0][1].matshow(X_train[11])
axes[0][2].matshow(X_train[12])
axes[0][3].matshow(X_train[13])
axes[0][4].matshow(X_train[14])

# 2행 데이터
axes[1][0].matshow(X_train[15])
axes[1][1].matshow(X_train[16])
axes[1][2].matshow(X_train[17])
axes[1][3].matshow(X_train[18])
axes[1][4].matshow(X_train[19])

#%%
# 데이터 전처리
## 신경망의 노드에 맞게 차원 변경 (28, 28) -> (784)
## 출력층의 Label 0, 1로 표현
# 3-> [ 0 0 0 1 0 0 0 0 0 0 0]
# 4-> [ 0 0 0 0 1 0 0 0 0 0 0]

import numpy as np

X_train_n =X_train.reshape(60000, 28*28)
X_test_n =X_test.reshape(10000, 28*28)

print(X_train_n.shape, X_test_n.shape)

print(y_train[0])
# eye 함수의 인풋으로 라벨의 갯수를 넣는다. 원핫 인코딩
print(np.eye(10)[y_train[0]])


#%%
y_train_onehot = np.eye(10)[y_train]
y_test_onehot = np.eye(10)[y_test]

#%%
# 03 활성화 함수 정의
# sigmoid 함수 만들기
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#%%
# linespace는 구간내에 들어가는 임의 값을 만드는 함수
x_data = np.linspace(start =-10, stop= +10)
print(x_data)
y_data = sigmoid(x_data)
plt.plot(x_data, y_data)
    
#%%
# 05. 학습시키기
num_epoch =300
learning_rate = 0.00001

w = np.random.uniform(low=0.0, high=1.0, size=(28*28, 10))
b = np.random.uniform(low=0.0, high=1.0, size=10)


# w, b값을 찾는다.
# y = wx + b
#%%

for epoch in range(num_epoch):
    y_pred_onehot = X_train_n.dot(w) +b # 60000 X 784 * 784 * 10
    y_pred_onehot = sigmoid(y_pred_onehot)
    y_pred = y_pred_onehot.argmax(axis=1)
    
    error = (y_train != y_pred).mean() # ==이면 정확도
    
    if error < 0.1:
        break
    
    if epoch % 10 == 0:
        print('{0} epoch, error={1:.5f}'.format(epoch, error))
        
    w = w - learning_rate * X_train_n.T.dot(y_pred_onehot-y_train_onehot)
    b = b - learning_rate * (y_pred_onehot-y_train_onehot).mean(axis=0)
    
print('{0}, {1}'.format(epoch, error))