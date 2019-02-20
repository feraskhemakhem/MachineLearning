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
        x1, x2 = sample[1], sample[2]
        poly.append([1, x1, x2, np.power(x1, 2), x1*x2, 
           np.power(x2,2), np.power(x1, 3), np.power(x1,2)*x2, x1*np.power(x2,2),
           np.power(x2, 3)])
        
    return np.array(poly)
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
        Ein = 0
        n_samples = len(X)
        n_features = len(X[0])
    
    
        for n in range(n_samples):
            # Ein = 1/N Sum to N (ln ( 1 + e ^ (-yn * wTn * xn)))
            Ein += np.dot(y[n],X[n]) / (1 + np.power(np.e, y[n]*np.matmul(X[n], self.W)))
        
        Ein /= float(n_samples)
        
        return -1*Ein
        
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
        # if we are working with a third order polynomial, transform
        size = 3
        self.W = []
        if self.third_order:
            X = third_order(X)
            size = 10
            
        # if normal, 3, if third order, 9
        for a in range(size):
            self.W.append(0)
            
        for a in range(0, self.max_iter):
            # get nv, the gradient

                gradient = self._gradient(X, y)
                
                # w(t+1) = w(t) + nv
                self.W -= self.lr * gradient
    			
    			# continue until || gradient || is 0
                if np.linalg.norm(gradient) is 0:
                    break

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
        if self.third_order:
            X = third_order(X)
    
        # probability
        preds = []
        for i in range(len(X)):
            if (1 / (1 + np.power(np.e, -1*(np.matmul(X[i], self.W))))) >= 0.5:
                preds.append(1)
            else:
                preds.append(-1)
        return preds

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