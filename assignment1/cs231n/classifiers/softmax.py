import numpy as np
from random import shuffle
from past.builtins import xrange

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
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    #correct_class_score = scores[y[i]]
    #num = np.exp(correct_class_score)
    scores_exp = np.exp(scores)
    denum = np.sum(scores_exp)
    invden = 1.0/denum
    f = scores_exp * invden
    loss += -np.log(f[y[i]])
    
    f[y[i]] -= 1
    
    for j in xrange(num_classes):
      dW[:,j] += X[i,:] * f[j]
    #loss += -np.log(np.exp(correct_class_score)/np.sum(np.exp(scores)))
    #dloss = -1/f
    #dnum = invden * dloss
    #dinvden = num * dloss
    #ddenum = (-1.0 / (denum**2)) * dinvden
    #dscores_exp = scores_exp * ddenum
    #dscores = np.exp(scores) * dscores_exp
    #temp = X[i].reshape(X[i].shape[0], 1)
    #dscores = dscores.reshape(dscores.shape[0], 1)
    #dW += temp.dot(dscores.T)
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_train
  dW += reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= np.max(scores,axis=1,keepdims=True)
  f = np.exp(scores) / np.sum(np.exp(scores),axis=1,keepdims=True)
  correct_class_probabilities = f[range(num_train),y]

  loss = np.sum(-np.log(correct_class_probabilities))/num_train
  loss += 0.5 * reg * np.sum(W*W)

  f[range(num_train),y] -=1
  dW = X.T.dot(f)/num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

