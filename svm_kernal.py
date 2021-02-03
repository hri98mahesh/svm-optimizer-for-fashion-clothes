import csv
import math
import numpy as np
import time
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import SVC

start_time = time.time()
with open("fashion_mnist/train.csv") as csv_file:
	# input_data =  np.zeros((0,0))
	entry_number = 5
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_x = []
	input_y = []
	for row in csv_reader:
		if(float(row[len(row)-1])==entry_number or float(row[len(row)-1])==entry_number+1):
			input_x.append([float(x)/255 for x in row[:len(row)-1]])
			input_y.append(float(row[len(row)-1])*2-2*entry_number-1)
	# print(input_y)
	datax = np.array(input_x)
	number_of_features = datax.shape[1]
	datax_temp = np.array(input_x)
	datay = np.array(input_y)
	datay_temp = np.array(input_y)
	number_of_training = datax.shape[0]
	# print(number_of_training)
	datay = np.reshape(datay,(datay.shape[0],1))
	datay = np.dot(datay,np.transpose(datay))
	datax_norm = np.sum(datax ** 2, axis = -1)
	gamma = 0.05
	datax = np.exp(-gamma * (datax_norm[:,None] + datax_norm[None,:] - 2 * np.dot(datax, np.transpose(datax))))
	P = matrix(datax*datay)
	q = matrix(-np.ones((number_of_training,1)))
	G = matrix(np.concatenate((np.diag(np.ones(shape=number_of_training))*(-1),np.diag(np.ones(shape=number_of_training))),axis =0))
	h = matrix(np.zeros(2*number_of_training))
	h[number_of_training:] = 1
	h = matrix(h)
	A = matrix(np.reshape(datay_temp,(1,number_of_training)))
	b = matrix(np.zeros(1))
	sol = solvers.qp(P,q,G,h,A,b)
	alphas = np.array(sol['x'])
	print(alphas)
	qw = np.ones((number_of_training,number_of_training))*np.multiply(np.reshape(datay_temp,(number_of_training,1)) , np.reshape(alphas,(number_of_training,1)))
	# print(qw)
	value = np.sum(datax * qw,axis = 0)
	# print(value)
	min_value  = []
	max_value  = []
	for i in range(0,number_of_training):
		if datay_temp[i] == -1:
			max_value.append(value[i])
		else:
			min_value.append(value[i])
	b = -(max(max_value)+min(min_value))/2.0
	print(b)

end_training_time = time.time()-start_time
print("The Training time is "+ str(end_training_time))
start_time = time.time()

with open("fashion_mnist/val.csv") as csv_file:
# 	entry_number = 5
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_x_test = []
	input_y_test = []
	for row in csv_reader:
		if(float(row[len(row)-1])==entry_number or float(row[len(row)-1])==entry_number+1):
			input_x_test.append([float(x)/255 for x in row[:len(row)-1]])
			input_y_test.append(float(row[len(row)-1])*2-2*entry_number-1)
	datax_test = np.array(input_x_test)
	datax_test_temp = np.array(input_x_test)
	datay_test = np.array(input_y_test)
	datay_test_temp = np.array(input_y_test)
	number_of_test = datax_test.shape[0]
	datax_norm_train = np.sum(datax_temp ** 2, axis = -1)
	datax_norm_test = np.sum(datax_test **2 ,axis = -1)
	gamma = 0.05
	datax_test = np.exp(-gamma * (datax_norm_train[:,None] + datax_norm_test[None,:] - 2 * np.dot(datax_temp, np.transpose(datax_test_temp))))
	# print(datax_test.shape)
	qw = np.ones((number_of_training,number_of_test))*np.multiply(np.reshape(datay_temp,(number_of_training,1)) , np.reshape(alphas,(number_of_training,1)))
	test_value = np.sum(datax_test*qw,axis = 0) + b
	print(test_value.shape)
	# print(value.shape)
	correct_count = 0.0
	total_count = 0.0
	for i in range(0,number_of_test):
		total_count += 1
		if((1==datay_test[i] and test_value[i]>=0) or (datay_test[i]==-1 and test_value[i]<0)):
			# print("the trained data comes out to be :")
			# print(datay_test[i])
			# print(test_value[i])
			correct_count +=1
	print(correct_count)
	print("The accuracy from the given data set comes out to be "+ str(100*float(correct_count/total_count))+"%")
	# print(np.transpose(w).shape)

end_testing_time = time.time()-start_time
print("The testing time is "+str(end_testing_time))