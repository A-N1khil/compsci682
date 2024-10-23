pass
from cs682.layers import *
from cs682.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

def affine_batchnorm_relu_forward(X, W, b, gamma, beta, param, normalization):
    """
    Convenience layer that performs an affine transform followed by a batchnorm and ReLU
    Inputs:
        X: Input to the affine layer
        W, b: Weights for the affine layer
        gamma, beta: Weights for the batchnorm layer
        param: Dictionary with parameters for the batchnorm layer
        normalization: Type of normalization to use
    Outputs:
        out: Output from the ReLU
        cache: Object to give to the backward pass
    """

    # First thing we do, is affine the forward pass
    a, fp_cache = affine_forward(X, W, b)

    bn, bn_cache = None, None

    # Next, we do the batchnorm forward pass
    if normalization == 'batchnorm':
        bn, bn_cache = batchnorm_forward(a, gamma, beta, param)
    elif normalization == 'layernorm':
        bn, bn_cache = layernorm_forward(a, gamma, beta, param)

    # Finally, we do the ReLU forward pass
    out, relu_cache = relu_forward(bn)

    # Store all the caches
    cache = (fp_cache, bn_cache, relu_cache)

    # Return the output and the cache
    return out, cache


def affine_batchnorm_relu_backward(dout, cache, normalization):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    Inputs:
        dout: Upstream derivative
        cache: Tuple of caches from the forward pass
        normalization: Type of normalization to use
    Outputs:
        dx, dw, db, dgamma, dbeta: Gradients with respect to the input, weights, biases, gamma and beta
    """

    # Unpack the caches
    fp_cache, bn_cache, relu_cache = cache

    # First, we do the ReLU backward pass
    drbp = relu_backward(dout, relu_cache)

    # Next, we do the batchnorm backward pass
    if normalization == 'batchnorm':
        dbn, dgamma, dbeta = batchnorm_backward_alt(drbp, bn_cache)
    elif normalization == 'layernorm':
        dbn, dgamma, dbeta = layernorm_backward(drbp, bn_cache)

    # Finally, we do the affine backward pass
    dx, dw, db = affine_backward(dbn, fp_cache)

    return dx, dw, db, dgamma, dbeta


