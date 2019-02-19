import numpy as np
import sys
from helper import *


def third_order(X):
    """Third order polynomial transform on features X.

    Args:
        X: An array with shape [n_samples, 2].

    Returns:
        poly: An (numpy) array with shape [n_samples, 10].
    """
	### YOUR CODE HERE
    poly= []
    for sample in X:
        x1, x2 = sample[0], sample[1]
        poly.append([1, x1, x2, np.power(x1, 2), x1*x2, 
           np.power(x2,2), np.power(x1,2)*x2, x1*np.power(x2,2),
           np.power(x2, 3)])
        
    return poly
    ### END YOUR CODE


class LogisticRegression(object):
    
    def __init__(self, max_iter, learning_rate, third_order=False):
        self.max_iter = max_iter
        self.lr = learning_rate
        self.third_order = third_order


    def _gradient(self, X, y):
        """Compute the gradient with samples (X, y) and weights self.W.

        Args:
            X: An array with shape [n_samples, n_features].
               (n_features depends on whether third_order is applied.)
            y: An array with shape [n_samples,]. Only contains 1 or -1.

        Returns:
            gradient: An array with shape [n_features,].
        """
        ### YOUR CODE HERE
        preds = predict(X)
        
        # partial derivates
        gradient = np.matmul(X, preds - y)
        
        
        gradient = gradient * self.lr / len(X)
        
       
        return gradient
        ### END YOUR CODE


    def fit(self, X, y):
        """Train logistic regression model on data (X,y).
        (If third_order is true, do the 3rd order polynomial transform)

        Args:
            X: An array with shape [n_samples, 3].
            y: An array with shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        self.W = np([0, 0, 0])
        for i in range(0, self.max_iter):
            # update weights and get gradient
            gradient = self._gradient(X, y)
            self.W -= gradient
            
            # predict
            preds = predict(X)
            
            # find error when y = 1
            err1 = -y*np.log(preds)
            
            # find error when y = -1
            err2 = (1-labels)*np.log(1-predictions)
            
            # take cost and apply
            cost = (err1 + err2) / len(y)


        ### END YOUR CODE
        return self


    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
               (n_features depends on whether third_order is applied.)
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.
        (If third_order is true, do the 3rd order polynomial transform)

        Args:
            X: An array of shape [n_samples, 3].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE
        
        #sign or sigmoid?
        return np.sign(np.matmul(X, self.W))

        ### END YOUR CODE


    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: A float. Mean accuracy of self.predict(X) wrt. y.
        """
        return np.mean(self.predict(X)==y)



def accuracy_logreg(max_iter, learning_rate, third_order, 
                    X_train, y_train, X_test, y_test):

    # train perceptron
    model = LogisticRegression(max_iter, learning_rate, third_order)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)

    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return train_acc, test_acc