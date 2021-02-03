import csv
import math
import numpy as np
import time
from cvxopt import matrix
from cvxopt import solvers

def train(class1,class2,input_data_class1,input_data_class2):
	datax = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datax_temp = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datay = np.ones(datax.shape[0])*(-1)
	datay[:len(input_data_class1)] += 2
	datay_temp = np.ones(datax.shape[0])*(-1)	
	datay_temp[:len(input_data_class1)] += 2
	number_of_training = datax.shape[0]
	number_of_features = datax.shape[1]
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
	qw = np.ones((number_of_training,number_of_training))*np.multiply(np.reshape(datay_temp,(number_of_training,1)) , np.reshape(alphas,(number_of_training,1)))
	value = np.sum(datax * qw,axis = 0)
	min_value  = []
	max_value  = []
	for i in range(0,number_of_training):
		if datay_temp[i] == -1:
			max_value.append(value[i])
		else:
			min_value.append(value[i])
	b = -(max(max_value)+min(min_value))/2.0
	return alphas,b

def test(datax_test,input_data_class1,input_data_class2,alpha,b,class_1,class_2):
	datax = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datax_temp = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datay = np.ones(datax.shape[0])*(-1)
	datay[:len(input_data_class1)] += 2
	datay_temp = np.ones(datax.shape[0])*(-1)	
	datay_temp[:len(input_data_class1)] += 2
	alpha = np.array(alpha)
	# print("the size of alpha is "+str(alpha.shape))
	number_of_training = datax.shape[0]
	datax_test = np.array(datax_test)
	number_of_test = datax_test.shape[0]
	datax_norm_train = np.sum(datax_temp ** 2, axis = -1)
	datax_norm_test = np.sum(datax_test **2 ,axis = -1)
	gamma = 0.05
	datax_test = np.exp(-gamma * (datax_norm_train[:,None] + datax_norm_test[None,:] - 2 * np.dot(datax_temp, np.transpose(datax_test))))
	qw = np.ones((number_of_training,number_of_test))*np.multiply(np.reshape(datay_temp,(number_of_training,1)) , np.reshape(alpha,(number_of_training,1)))
	test_value = np.sum(datax_test*qw,axis = 0) + b
	result = []
	for i in range(0,test_value.shape[0]):
		if test_value[i] >=0:
			result.append(class_1)
		else:
			result.append(class_2)
	return result


start_time = time.time()
with open("fashion_mnist/train.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_data = [[],[],[],[],[],[],[],[],[],[]]
	input_total = np.zeros(10)
	for row in csv_reader:
		class_data = float(row[len(row)-1])
		input_data[int(class_data)].append([float(x)/255 for x in row[:len(row)-1]])
		input_total[int(class_data)] += 1
	alphas = []
	b = []
	for i  in range(0,10):
		for j in range(i+1,10):
			print(str(i)+" "+str(j))
			temp_alphas,temp_b = train(i,j,input_data[i],input_data[j])
			alphas.append(temp_alphas)
			b.append(temp_b)

end_training_time = time.time()-start_time
print("The Training time is "+ str(end_training_time))
# start_time = time.time()

with open("fashion_mnist/val.csv") as csv_file:
	csv_reader = csv.reader(csv_file,delimiter=',')
	input_x_test = []
	input_y_test = []
	for row in csv_reader:
		class_data = float(row[len(row)-1])
		input_x_test.append([float(x)/255 for x in row[:len(row)-1]])
		input_y_test.append(int(class_data))
	count =0
	result = []
	for i in range(0,10):
		for j in range(i+1,10):
			temp_result = test(input_x_test,input_data[i],input_data[j],alphas[count],b[count],i,j)
			result.append(temp_result)
			count += 1

result1 = np.array(result)
resultT = np.transpose(result1)
correct_count = 0.0
total_count = 0.0
result_class = []
for i in range(0,resultT.shape[0]):
	ch = np.zeros(10)
	for j in range(0,resultT.shape[1]):
		ch[resultT[i][j]] += 1
	index = 0 
	max_v = 0
	for j in range(0,10):
		if(ch[j]>max_v):
			index = j
			max_v = ch[j]
	result_class.append(index)
	if(index == input_y_test[i]):
		correct_count += 1
	total_count += 1


print(correct_count)
# 4253.0
print("The total accuracy is "+ str(100*correct_count/total_count)+"%")
# The total accuracy is 85.06%

confusion_matrix = np.zeros((10,10))
for i in range(0,resultT.shape[0]):
	confusion_matrix[input_y_test[i]][result_class[i]] += 1
print(confusion_matrix)
# [405.,   0.,   0.,  18.,   0.,   1.,  56.,   0.,   1.,   0.]
# [  0., 484.,   0.,  10.,   1.,   0.,   1.,   0.,   0.,   0.]
# [  7.,   6., 414.,   2.,  56.,   0.,  57.,   0.,   1.,   0.]
# [  7.,   2.,   4., 412.,  14.,   0.,   6.,   0.,   0.,   0.]
# [  0.,   0.,  26.,   6., 365.,   0.,  20.,   0.,   0.,   0.]
# [  0.,   0.,   0.,   0.,   0., 436.,   0.,  50.,   1.,   5.]
# [ 71.,   7.,  43.,  42.,  52.,   0., 345.,   0.,   3.,   0.]
# [  0.,   0.,   0.,   0.,   0.,   7.,   0., 412.,   0.,   6.]
# [ 10.,   1.,  13.,  10.,  12.,  44.,  15.,   4., 494.,   3.]
# [  0.,   0.,   0.,   0.,   0.,  12.,   0.,  34.,   0., 486.]


