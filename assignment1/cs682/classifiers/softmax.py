import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # 1TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):

    # compute scores
    scores_i = X[i].dot(W)

    # normalize scores to prevent numerical instability
    scores_i -= np.max(scores_i)

    exponent_scores_i = np.exp(scores_i)
    exponent_scores_sum_i = np.sum(exponent_scores_i)

    # compute loss
    prob = exponent_scores_i / exponent_scores_sum_i

    # Add the loss of the correct class => -log(prob[y[i]])
    # Since it is - log, we can term loss = loss + (-log(prog[correct_class])) = loss - log(prob[correct_class])
    loss -= np.log(prob[y[i]])

    # Gradient for the correct class
    dW[:, y[i]] += (prob[y[i]] - 1) * X[i]

    # Compute the gradient
    for j in range(num_classes):
      if j == y[i]:
        # We have already computed the gradient for the correct class
        continue
      dW[:, j] += prob[j] * X[i]

  # Average the loss
  loss /= num_train

  # Add regularization to the loss
  loss += reg * np.sum(W * W)

  # Average the gradient
  dW /= num_train

  # Add regularization to the gradient
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  # Compute the scores
  scores = X.dot(W)

  # Subtract the maximum score from each row to prevent numerical instability
  scores -= np.max(scores, axis=1, keepdims=True)

  # Softmax loss calculation
  exponent_scores = np.exp(scores)
  sum_exponent_scores = np.sum(exponent_scores, axis=1, keepdims=True)

  # Compute the probabilities
  probs = exponent_scores / sum_exponent_scores

  # Computing the loss
  correct_class_probs = probs[np.arange(X.shape[0]), y]
  loss = np.sum(-np.log(correct_class_probs))

  # Average the loss
  loss /= X.shape[0]

  # Add regularization to the loss
  loss += reg * np.sum(W * W)

  # Compute the gradient

  # Gradient for the correct class
  probs[np.arange(X.shape[0]), y] -= 1

  dW = X.T.dot(probs)

  # Average the gradient
  dW /= X.shape[0]

  # Add regularization to the gradient
  dW += 2 * reg * W

  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

