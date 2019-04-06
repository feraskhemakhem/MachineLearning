import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework4: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different value 
	of cost c to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The value of cost c you need to try is listing as follow:
	c = [0.01, 0.1, 1, 2, 3, 5]
	Please keep other parameter options as default.
	No return value is needed
	'''
	c = [0.01, 0.1, 1, 2, 3, 5]
	for i in c:
		model = svm_train(train_label, train_data, '-c ' + str(i))
		svm_predict(test_label, test_data, model)

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different kernel
	to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The kernel  you need to try is listing as follow:
	0. linear kernel
	1. polynomial kernel
	2. radial basis function kernel
	Please keep other parameter options as default.
	No return value is needed
	'''
	for i in range(3):
		model = svm_train(train_label, train_data, '-t ' + str(i))
		svm_predict(test_label, test_data, model)
