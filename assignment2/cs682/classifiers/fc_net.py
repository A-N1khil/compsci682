from builtins import range
from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO1: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        # For a Gaussian distribution, we can use np.random.normal(mean, std_dev, size)

        # W1 should be of the dimensions (D, H) => (input_dim, hidden_dim)
        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))

        # b1 should be of the dimensions (H,) => (hidden_dim,)
        self.params['b1'] = np.zeros(hidden_dim)

        # W2 should be of the dimensions (H, C) => (hidden_dim, num_classes)
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))

        # b2 should be of the dimensions (C,) => (num_classes,)
        self.params['b2'] = np.zeros(num_classes)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO1: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        # Retrieve the params
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        # Perform the forward pass. Since we have already implemented the affine_relu_forward, we can simply use that
        # Use affine_relu_forward for the first layer because it affines the input and then applies the ReLU activation
        A1, cache1 = affine_relu_forward(X, W1, b1)

        # Use affine_forward for the second layer because it only affines the input
        scores, cache2 = affine_forward(A1, W2, b2)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO2: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # Softmax loss is already computed in the layers.py file. We can use the softmax_loss function from there
        loss, dscores = softmax_loss(scores, y) # We have the loss as well as the gradient of the loss wrt the scores

        # Add the regularization term to the loss
        reg_term = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss += reg_term

        # Compute the gradient of the loss wrt the scores
        # Use affine_backward for the second layer because it only affines the input
        dA1, dW2, db2 = affine_backward(dscores, cache2)

        # Use affine_relu_backward for the first layer because it affines the input and then applies the ReLU activation
        dX, dW1, db1 = affine_relu_backward(dA1, cache1)

        # Add the regularization term to the gradients
        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO1: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################

        # Note to self:
        # W1 should be of the dimensions (D, H1) => (input_dim, hidden_dim[0])
        # b1 should be of the dimensions (H1,) => (hidden_dim[0],)
        # W2 should be of the dimensions (H1, H2) => (hidden_dim[0], hidden_dim[1])
        # b2 should be of the dimensions (H2,) => (hidden_dim[1],)
        # This would mean that we will have to add special cases for the first and last layers. Which can be handled if used np.hstack

        dimensions_arr = np.hstack((input_dim, hidden_dims, num_classes)) # Will give an array of dimensions [input_dim, hidden_dim[0], hidden_dim[1], ..., num_classes]

        # Now, we can iterate over the dimensions_arr to initialize the weights and biases
        for index in range(self.num_layers):
            # For, i=0 (first layer), W1 should be of the dimensions (D, H1) => (input_dim, hidden_dim[0]) => dimensions_arr[0], dimensions_arr[1]
            # For, i=1 (second layer), W2 should be of the dimensions (H1, H2) => (hidden_dim[0], hidden_dim[1]) => dimensions_arr[1], dimensions_arr[2]
            # For the last layer, W3 should be of the dimensions (H2, C) => (hidden_dim[1], num_classes) => dimensions_arr[-2], dimensions_arr[-1]
            self.params[f"W{index + 1}"] = np.random.randn(dimensions_arr[index], dimensions_arr[index + 1]) * weight_scale

            # For, i=0 (first layer), b1 should be of the dimensions (H1,) => (hidden_dim[0],) => dimensions_arr[1]
            # For, i=1 (second layer), b2 should be of the dimensions (H2,) => (hidden_dim[1],) => dimensions_arr[2]
            # For the last layer, b3 should be of the dimensions (C,) => (num_classes,) => dimensions_arr[-1]
            self.params[f"b{index + 1}"] = np.zeros(dimensions_arr[index + 1])

            # If we are using batch normalization, we will have to initialize the gamma and beta parameters
            # Normalization is not used for the last layer, hence we will not initialize the gamma and beta for the last layer
            if self.normalization and index != self.num_layers - 1:
                # Scale params to be initialized to ones
                self.params[f"gamma{index + 1}"] = np.ones(dimensions_arr[index + 1])
                # Shift params to be initialized to zeros
                self.params[f"beta{index + 1}"] = np.zeros(dimensions_arr[index + 1])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        # Creating a cache list to store the cache for each layer
        caches = {}

        # We will iterate over the all the layers from first to n-1
        for layer in range(self.num_layers - 1):
            # Extracting the parameters for the layer
            W = self.params[f"W{layer + 1}"]
            b = self.params[f"b{layer + 1}"]

            if self.normalization:
                # If the normalization is batchnorm, we will use the affine_bn_relu_forward function
                # Extract the gamma and beta parameters
                gamma = self.params[f"gamma{layer + 1}"]
                beta = self.params[f"beta{layer + 1}"]

                # Defining a function to avoid complicating code here
                # NOTE TO SELF:
                # As stated in the note above, have to pass self.bn_params[0] to the forward pass for the first batch normalization layer
                # and pass self.bn_params[1] to the forward pass for the second batch normalization layer
                # thus effectively making the index of the bn_params list as the layer number
                X, caches[layer] = affine_batchnorm_relu_forward(X, W, b, gamma, beta, self.bn_params[layer], self.normalization)
            else:
                # We can use the affine_relu_forward and affine_forward functions from the layer_utils.py file
                X, caches[layer] = affine_relu_forward(X, W, b)

            # If we are using dropout, we will have to pass the 
            # Note to self: The dp_cache will be stored in the caches dictionary with the key as dropout{layer + 1}
            # Hence, extracting this would be different than simply extracting the layer cache
            if self.use_dropout:
                X, dp_cache = dropout_forward(X, self.dropout_param)
                caches[f'dropout{layer + 1}'] = dp_cache

        # For the last layer, we will use the affine_forward function
        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]

        # We need not use the batchnorm for the last layer
        scores, caches[self.num_layers - 1] = affine_forward(X, W, b)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dscores = softmax_loss(scores, y)

        # Adding regularization loss across all layers
        for layer in range(self.num_layers):
            W = self.params[f"W{layer + 1}"]
            reg_term = 0.5 * self.reg * np.sum(W * W)
            loss += reg_term

        # For the last layer, we will simply use the affine_backward function
        # Remember that we have indexed the caches from 0, so the last layer cache will be caches[self.num_layers - 1]
        # BN edit: We need not use the batchnorm for the last layer
        dout, dw, db = affine_backward(dscores, caches[self.num_layers - 1])

        # Save these to grads
        grads[f"W{self.num_layers}"] = dw + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        # Now, we will iterate over the layers from n-1 to 0
        # Start off with the penultimate layer and keep decreasing by 1 until we reach 0
        # Since we have indexed the caches from 0, the penultimate layer will be self.num_layers - 2

        # BN edit: I just realised I can simply use a reverse range iterator here instead of coding like 10 year old
        for layer in reversed(range(self.num_layers - 1)):
            # Extract the cache for the layer
            cache = caches[layer]

            # If we are using dropout, then we will have to perform the back prop for dropout
            if self.use_dropout:
                dout = dropout_backward(dout, caches[f'dropout{layer + 1}'])

            if self.normalization:
                dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(dout, cache, self.normalization)

                # Save dgama and dbeta to grads
                grads[f"gamma{layer + 1}"] = dgamma
                grads[f"beta{layer + 1}"] = dbeta
            else:
                # Compute the gradients like we did previously, but this time we use the affine_relu_backward function
                dout, dW, db = affine_relu_backward(dout, cache)

            # Save these to grads
            grads[f"W{layer + 1}"] = dW + self.reg * self.params[f"W{layer + 1}"]
            grads[f"b{layer + 1}"] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
