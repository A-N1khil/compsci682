import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i] # decrease the gradient of the incorrect class
        dW[:, y[i]] -= X[i] # increase the gradient of the correct class

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # 1TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  # Average the gradient over all training examples
  dW /= num_train

  # Add regularization to the gradient
  dW += 2 * reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # 1TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # Compute the scores
  scores = X.dot(W)

  # Correct class scores
  num_train = len(y)
  correct_class_scores = scores[np.arange(num_train), y]

  # Add a new axis to the correct class scores to allow broadcasting
  correct_class_scores = correct_class_scores[:, np.newaxis]

  # Compute the margins
  margins = np.maximum(0, scores - correct_class_scores + 1)

  # Remove the margin for the correct class
  margins[np.arange(num_train), y] = 0

  loss = np.sum(np.fmax(margins, 0)) / num_train  # Average the loss over all training examples

  # Add regularization to the loss
  loss += reg * np.sum(W * W)

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # [DoneTODO]:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  # Assume a binary matrix M where B[i, j] = True if margin > 0, False otherwise
  binary_matrix = (margins > 0).astype(int)

  # Calculate the sum of all the positive margins for each training example
  row_sum = np.sum(binary_matrix, axis=1)

  # Subtract the number of positive margins for the correct class from the binary matrix
  binary_matrix[np.arange(num_train), y] = -row_sum

  # Compute the gradient
  dW = X.T.dot(binary_matrix)

  # Average the gradient over all training examples
  dW /= num_train

  # Add regularization to the gradient
  dW += 2 * reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
