import numpy as np
import warnings

warnings.filterwarnings('ignore')
'''
source : https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
'''

def im2col(image, _filter, stride=1):
    n_c, h, w = image.shape
    _, _h, _w = _filter.shape
    
    out_h = (h - _h)//stride + 1
    out_w = (w - _w)//stride + 1
    
    im_col = np.zeros([_h*_w*n_c, out_h*out_w])
    flt_col = _filter.reshape(n_c*_h*_w, -1)
    
    for i in range(out_h):
        for j in range(out_w):
            im_col[:, i*out_w+j] = image[:, i*stride:i*stride+_h, j*stride:j*stride+_w].reshape(-1)
            
    return im_col, flt_col

def convolution(image, filt, bias, s=1):
    n_f, n_c_f, f, _ = filt.shape
    n_c, in_dim, _ = image.shape

    out_dim = int((in_dim - f) / s) + 1
    out = np.zeros((n_f, out_dim, out_dim))
    
    for current_filter in range(n_f):
        im_col, flt_col = im2col(image, filt[current_filter], s)
        out[current_filter] = np.reshape(np.dot(flt_col.T, im_col) + bias[current_filter], (out_dim, out_dim))
        
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
    n_f, n_c_f, f, _ = filt.shape
    n_c, in_dim, _ = conv_in.shape

    dbias = np.zeros((n_f, 1))
    dfilt = np.zeros(filt.shape)
    dout = np.zeros(conv_in.shape)
    
    # Backpropagation for each filter
    for current_filter in range(n_f):
        # Backpropagation for each position on the dconv_prev (convoluted matrix)
        for i in range(dconv_prev.shape[1]):
            for j in range(dconv_prev.shape[2]):
                dfilt[current_filter] += dconv_prev[current_filter, i, j] * conv_in[:, i*s:i*s+f, j*s:j*s+f]
                dout[:, i*s:i*s+f, j*s:j*s+f] += dconv_prev[current_filter, i, j] * filt[current_filter]

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







    