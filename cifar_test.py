from cifar_model import *
import numpy as np
import pickle
import time
import os

def test_data_loader(file):
    dataset = np.load(file)
    data = dataset['data'][:, :]
    labels = dataset['labels'].reshape(len(data), 1)

    dataset.close()
    
    data = data.astype('float32')
    # Centering
    data -= np.mean(data, axis=0)

    # Standardization
    std_dev = np.std(data, axis=0)
    std_dev[std_dev==0] = 1.0
    data /= std_dev

    return data, labels

def test(params, test_data, test_label, num_class):
    T = 0
    F = 0
    [f1, w2, w3, b1, b2, b3] = params

    images = test_data
    labels = test_label
    print(len(images))
    start = time.time()
    for i in range(len(test_data)):
        image = images[i].reshape(3, 32, 32)
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
    params_PATH = os.path.join(params_PATH, 'cifar_params.pkl')
    with open(params_PATH, 'rb') as file:
        params = pickle.load(file)
    num_classes = 10

    PATH = os.path.dirname(__file__)
    PATH = os.path.join(PATH, 'data', 'cifar_test.npz')
    test_data, test_label = test_data_loader(PATH)

    accuracy = test(params, test_data, test_label, num_classes)
    print("accuracy: ", accuracy)