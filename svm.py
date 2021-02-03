import csv
import math
import numpy as np
import time
from cvxopt import matrix
from cvxopt import solvers

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
	datax = np.array(input_x)
	number_of_features = datax.shape[1]
	datax_temp = np.array(input_x)
	datay = np.array(input_y)
	datay_temp = np.array(input_y)
	number_of_training = datax.shape[0]
	print(number_of_training)
	datay = np.reshape(datay,(datay.shape[0],1))
	datay = np.dot(datay,np.transpose(datay))
	datax = np.dot(datax,np.transpose(datax))
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
	# print(alphas)
	qw = np.multiply(np.reshape(datay_temp,(number_of_training,1)) , np.reshape(alphas,(number_of_training,1)))
	qw = np.ones((number_of_training,number_of_features))*qw
	w = np.sum(np.multiply(qw,datax_temp),axis =0)
	w = np.reshape(w,(1,w.shape[0]))
	# print(w)
	value = np.dot(w,np.transpose(datax_temp))
	min_value  = []
	max_value  = []
	for i in range(0,number_of_training):
		if datay_temp[i] == -1:
			max_value.append(value[0][i])
		else:
			min_value.append(value[0][i])

	b = -(max(max_value)+min(min_value))/2.0
	print(b)
end_training_time = time.time()-start_time
print("The Training time is "+ str(end_training_time))
start_time = time.time()
with open("fashion_mnist/test.csv") as csv_file:
	entry_number = 5
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_x_test = []
	input_y_test = []
	for row in csv_reader:
		if(float(row[len(row)-1])==entry_number or float(row[len(row)-1])==entry_number+1):
			input_x_test.append([float(x)/256 for x in row[:len(row)-1]])
			input_y_test.append(float(row[len(row)-1])*2-2*entry_number-1)
	datax_test = np.array(input_x_test)
	datax_test_temp = np.array(input_x_test)
	datay_test = np.array(input_y_test)
	datay_test_temp = np.array(input_y_test)
	number_of_test = datax_test.shape[0]
	# print(datay_test.shape)
	test_value = np.dot(w,np.transpose(datax_test)) + b
	correct_count = 0.0
	total_count = 0.0
	# print(datay_test.shape)
	# print(test_value.shape)
	for i in range(0,number_of_test):
		total_count += 1
		if((1==datay_test[i] and test_value[0][i]>=0) or (datay_test[i]==-1 and test_value[0][i]<0)):
			correct_count +=1
	print(correct_count)
	print("The accuracy from the given data set comes out to be "+ str(100*float(correct_count/total_count))+"%")
	# print(np.transpose(w).shape)
end_testing_time = time.time()-start_time
print("The testing time is "+str(end_testing_time))