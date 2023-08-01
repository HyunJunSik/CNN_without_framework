import numpy as np
import pandas as pd
import os
import pickle
import time
from mnist_model import *

PATH = os.path.dirname(__file__)
PATH = os.path.join(PATH, 'data', 'mnist_train.csv')

def train_data_loader(file):
    # read csv file
    raw_data = pd.read_csv(file)
    (rows, columns) = raw_data.iloc[:40000, :].shape
    # 40000개만 뽑아서 학습

    train_data = raw_data.iloc[:40000, 1:].values
    train_data = train_data.astype('float32')
    train_data = train_data/255.0

    train_label = raw_data.iloc[:40000, 0].values.reshape(rows, 1)
    
    return train_data, train_label, rows, int(np.sqrt(columns))

# filter 및 weight 초기화
def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

# Optimizer Adam 
def Adam(batchs, classes, learning_rate, dim, channels, beta1, beta2, params, cost):

    [f1, w2, w3, b1, b2, b3] = params

    df1, dw2, dw3, db1, db2, db3 = np.zeros(f1.shape), np.zeros(w2.shape), np.zeros(w3.shape), np.zeros(b1.shape), np.zeros(b2.shape), np.zeros(b3.shape)
    v1, v2, v3, s1, s2, s3 = np.zeros(f1.shape), np.zeros(w2.shape), np.zeros(w3.shape), np.zeros(f1.shape), np.zeros(w2.shape), np.zeros(w3.shape)
    bv1, bv2, bv3, bs1, bs2, bs3 = np.zeros(b1.shape), np.zeros(b2.shape), np.zeros(b3.shape), np.zeros(b1.shape), np.zeros(b2.shape), np.zeros(b3.shape), 
    
    X = batchs[:, 0:-1]
    X = X.reshape(len(batchs), channels, dim, dim)
    Y = batchs[:, -1]

    cost_ = 0
    batch_size = len(batchs)
    start = time.time()
    for b in range(batch_size):
        x = X[b]
        # one-hot encoding
        y = np.eye(classes)[int(Y[b])].reshape(classes, 1)
        grads, loss = model(x, y, params)
        [d_f1, d_w2, d_w3, d_b1, d_b2, d_b3] = grads
        df1 += d_f1
        dw2 += d_w2
        dw3 += d_w3
        db1 += d_b1
        db2 += d_b2
        db3 += d_b3
        cost_ += loss
    end = time.time()
    print(f"{end - start:.5f} sec")
    # Parameter Update by Adam
    ########################################################
    # first convoluton layer -> momentum optimizer 
    v1 = beta1 * v1 + (1-beta1) * df1 / batch_size
    bv1 = beta1 * bv1 + (1-beta1) * db1 / batch_size
    # first convolution layer -> RMSProp optimizer
    s1 = beta2 * s1 + (1-beta2) * (df1 / batch_size) ** 2
    bs1 = beta2 * bs1 + (1-beta2) * (db1 / batch_size) ** 2 
    # combine momentum and RMSProp -> Adam
    f1 -= learning_rate * v1 / np.sqrt(s1 + 1e-7)
    b1 -= learning_rate * bv1 / np.sqrt(bs1 + 1e-7)

    # first dense layer -> momentum optimizer
    v2 = beta1 * v2 + (1-beta1) * dw2 / batch_size
    bv2 = beta1 * bv2 + (1-beta1) * db2 / batch_size
    # first dense layer -> RMSProp optimizer
    s2 = beta2 * s2 + (1-beta2) * (dw2 / batch_size) ** 2
    bs2 = beta2 * bs2 + (1-beta2) * (db2 / batch_size) ** 2
    # combine momentum and RMSProp -> Adam
    w2 -= learning_rate * v2 / np.sqrt(s2 + 1e-7)
    b2 -= learning_rate * bv2 / np.sqrt(bs2 + 1e-7)

    # second dense layer
    v3 = beta1 * v3 + (1-beta1) * dw3 / batch_size
    bv3 = beta1 * bv3 + (1-beta1) * db3 / batch_size
    s3 = beta2 * s3 + (1-beta2) * (dw3 / batch_size) ** 2
    bs3 = beta2 * bs3 + (1-beta2) * (db3 / batch_size) ** 2
    w3 -= learning_rate * v3 / np.sqrt(s3 + 1e-7)
    b3 -= learning_rate * bv3 / np.sqrt(bs3 + 1e-7)
    ########################################################

    cost_ = cost_ / batch_size
    cost.append(cost_)
    print("loss :", sum(cost) / len(cost))
    params = [f1, w2, w3, b1, b2, b3]
    
    return params, cost



train_data, train_label, rows, image_size = train_data_loader(PATH)
dataset = np.hstack((train_data, train_label))

num_epochs = 10
batch_size = 512
num_filters = 4
image_channel = 1
learning_rate = 0.01
num_classes = 10
window_size = 8

beta1 = 0.95
beta2 = 0.99

f1 = (num_filters, image_channel, window_size, window_size)
# first dense layer nodes : 128
w2 = (128, 400)
# second dense layer nodes : 10
w3 = (10, 128)

f1 = initializeFilter(f1)
w2 = initializeWeight(w2)
w3 = initializeWeight(w3)

b1 = np.zeros((f1.shape[0], 1))
b2 = np.zeros((w2.shape[0], 1))
b3 = np.zeros((w3.shape[0], 1))

params = [f1, w2, w3, b1, b2, b3]
cost = []

print("Total data : ", len(train_data))
for epoch in range(num_epochs):
    np.random.shuffle(dataset)
    print("epoch :", epoch)
    cnt = 1
    for b in range(0, rows, batch_size):
        batchs = dataset[b : b + batch_size]
        params, cost = Adam(batchs, num_classes, learning_rate, image_size, image_channel, beta1, beta2, params, cost)
        print("batch", cnt, "/", int(len(train_data) / batch_size))
        cnt += 1

to_save = params

with open("mnist_params.pkl", 'wb') as file:
    pickle.dump(to_save, file)
