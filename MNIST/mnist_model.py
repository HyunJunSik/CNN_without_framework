import numpy as np
import warnings

warnings.filterwarnings('ignore')
'''
source : https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
'''

def convolution(image, filt, bias, s=1):
    '''
    convoltion 연산 함수(패딩 안넣음)
    ex) input size : (1, 28, 28) 
        output : (number of filter, (28 - sliding window height+ 1) / stride, (28 - sliding window width + 1) / stride)
        
    notation 
    filt : filter
    image : input image(mnist : (1, 28, 28)) 점점 크기는 줄어들것임
    s : stride
    '''

    n_f, n_c_f, f, _ = filt.shape
    # n_f : number of filter
    # n_c_f : number of channel of filter
    # f : window size

    n_c, in_dim, _ = image.shape
    # n_c : number of channel
    # in_dim : input image

    # output dimension calculate
    out_dim = int((in_dim - f) / s) + 1

    out = np.zeros((n_f, out_dim, out_dim))

    # convolution each filter over image
    for current_filter in range(n_f):
        current_y = out_y = 0
        # move across column of image matrix 
        while current_y + f <= in_dim:
            current_x = out_x = 0
            # move across row of image matrix
            while current_x + f <= in_dim:
                # add bias
                out[current_filter, out_y, out_x] = np.sum(filt[current_filter] * image[:, current_y : current_y + f, current_x : current_x + f]) + bias[current_filter]
                current_x += s
                out_x += 1
            current_y += s
            out_y += 1
    
    return out

def maxpooling(image, f=2, s=2):
    '''
    This function will dowmsample input image by using filter size of f and stirde of s
    '''
    
    n_c, h_prev, w_prev = image.shape

    # calculate output dimensions after maxpooling
    h = int((h_prev - f) / s) + 1
    w = int((w_prev - f) / s) + 1

    # before maxpooling operation, get maxpooling operation matrix

    downsampled = np.zeros((n_c, h, w))

    # Each channels will be operated by maxpooling
    for channel in range(n_c):
        current_y = out_y = 0
        # sliding max pooling window across column of image matrix
        while current_y + f <= h_prev:
            current_x = out_x = 0
            # sliding max pooling window across row of image matrix
            while current_x + f <= w_prev:
                # choose maximum value in window and store in output matrix
                downsampled[channel, out_y, out_x] = np.max(image[channel, current_y : current_y + f, current_x : current_x + f])
                current_x += s
                out_x += 1
            current_y += s
            out_y += 1
            
    return downsampled

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categorical_cross_entropy(probs, label):
    return -np.sum(label * np.log(probs))

def convolution_backward(dconv_prev, conv_in, filt, s=1):
    '''
    BackPropagation through a Convolutional layer. 
    '''

    (n_f, _, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    for current_filter in range(n_f):
        # loop every filters
        current_y = out_y = 0
        while current_y + f <= orig_dim:
            current_x = out_x = 0
            while current_x + f <= orig_dim:
                # loss gradient of filter
                dfilt[current_filter] += dconv_prev[current_filter, out_y, out_x] * conv_in[:, current_y : current_y + f, current_x : current_x + f]
                # loss gradient of the input to the convolution operation
                dout[:, current_y : current_y + f, current_x : current_x + f] += dconv_prev[current_filter, out_y, out_x] * filt[current_filter]
                current_x += s
                out_x += 1
            current_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[current_filter] = np.sum(dconv_prev[current_filter])

    return dout, dfilt, dbias

def nanargmax(arr):
    '''
    return index of the largest non-nan value in the array.
    Output is an ordered pair tuple
    '''
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)

    return idxs

def maxpooling_backward(dpool, orig, f=2, s=2):
    '''
    Backpropagation through a Maxpooling Layer
    Gradients are passed through the indices(index 복수) of greatest value in the original maxpooling
    '''
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for channel in range(n_c):
        current_y = out_y = 0
        while current_y + f <= orig_dim:
            current_x = out_x = 0
            while current_x + f <= orig_dim:
                # obtain indes of largest value in input for current window
                (y, x) = nanargmax(orig[channel, current_y : current_y + f, current_x : current_x + f])
                dout[channel, current_y + y, current_x + x] = dpool[channel, out_y, out_x]
                current_x += s
                out_x += 1
            current_y += s
            out_y += 1
    return dout

def model(image, label, params):
    
    # I'll stack two convolution layers and two dense layers
    [f1, w2, w3, b1, b2, b3] = params
    
    ## forward ##
    ########################################################
    # first layer of convolution
    conv1 = convolution(image, f1, b1)
    # ReLU non-linearity
    conv1[conv1<=0] = 0

    # MaxPooling layer
    pooling = maxpooling(conv1)
    (channels, height, width) = pooling.shape

    fully_connected = pooling.reshape((channels * height * width, 1))

    # first layer of dense
    dense = w2.dot(fully_connected) + b2
    # ReLU non-linearity
    dense[dense<=0] = 0

    # second layer of dense
    output = w3.dot(dense) + b3
    
    # predict class probabilites
    probs = softmax(output)

    # calculate loss value
    loss = categorical_cross_entropy(probs, label)
    ########################################################

    ## backward ##
    ########################################################
    # categorical cross entropy derivative
    d_output = probs - label

    # second layer of dense derivative
    # d_w4
    d_w3 = d_output.dot(dense.T)
    # d_b4
    d_b3 = np.sum(d_output, axis=1).reshape(b3.shape)
    
    # first layer of dense derivative
    d_dense = w3.T.dot(d_output)
    # ReLU non-linearity
    d_dense[dense<=0] = 0
    
    # d_w3
    d_w2 = d_dense.dot(fully_connected.T)
    d_b2 = np.sum(d_dense, axis=1).reshape(b2.shape)

    # MaxPooling derivative
    d_fully_connected = w2.T.dot(d_dense)
    d_pooling = d_fully_connected.reshape(pooling.shape)

    d_conv1 = maxpooling_backward(d_pooling, conv1)
    
    d_conv1[conv1<=0] = 0
    # first layer of convolution derivative
    d_image, d_f1, d_b1 = convolution_backward(d_conv1, image, f1)
    ########################################################

    grads = [d_f1, d_w2, d_w3, d_b1, d_b2, d_b3]

    return grads, loss







    