from mnist_model import *
import numpy as np
import pickle
import time
import pandas as pd
import os

def test_data_loader(file):
    raw_data = pd.read_csv(file)
    (rows, columns) = raw_data.iloc[:, :].shape

    test_data = raw_data.iloc[:, 1:].values
    test_label = raw_data.iloc[:, 0].values.reshape(rows, 1)

    return test_data, test_label

def test(params, test_data, test_label, num_class):
    T = 0
    F = 0
    [f1, w2, w3, b1, b2, b3] = params

    images = test_data
    labels = test_label
    print(len(images))
    start = time.time()
    for i in range(len(test_data)):
        image = images[i].reshape(1, 28, 28)
        label = np.eye(num_class)[int(labels[i])].reshape(num_class, 1)
        conv1 = convolution(image, f1, b1)
        conv1[conv1<=0] = 0
        pooling = maxpooling(conv1)
        (channels, height, width) = pooling.shape
        fully_connected = pooling.reshape((channels * height * width, 1))
        dense = w2.dot(fully_connected) + b2
        dense[dense<=0] = 0
        output = w3.dot(dense) + b3
        probs = softmax(output)

        # accuracy 측정 
        if np.argmax(probs) == np.argmax(label):
            T += 1
        else:
            F += 1
            
    end = time.time()
    print(f"{end - start:.5f} sec")
    accuracy = T / (T + F)
    return accuracy

if __name__ == "__main__":

    params_PATH = os.path.dirname(__file__)
    params_PATH = os.path.join(params_PATH, 'mnist_params.pkl')
    with open(params_PATH, 'rb') as file:
        params = pickle.load(file)
    num_classes = 10

    PATH = os.path.dirname(__file__)
    PATH = os.path.join(PATH, 'data', 'mnist_test.csv')
    test_data, test_label = test_data_loader(PATH)

    accuracy = test(params, test_data, test_label, num_classes)
    print("accuracy: ", accuracy)