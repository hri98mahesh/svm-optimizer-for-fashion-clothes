import csv
import math
import numpy as np
from cvxopt import matrix
import time
from cvxopt import solvers
from sklearn.svm import SVC

start_time = time.time()

with open("fashion_mnist/train.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_x = []
	input_y = []
	for row in csv_reader:
		input_x.append([float(x)/255 for x in row[:len(row)-1]])
		input_y.append(float(row[len(row)-1]))
	datax = np.array(input_x)
	number_of_features = datax.shape[1]
	datax_temp = np.array(input_x)
	datay = np.array(input_y)
	datay_temp = np.array(input_y)
	number_of_training = datax.shape[0]
	clf1 = SVC(kernel='rbf',gamma=0.05)
	clf1.fit(datax,datay)

end_training_time = time.time()-start_time
print("The Training time is "+ str(end_training_time))
start_time = time.time()

with open("fashion_mnist/test.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_x_test = []
	input_y_test = []
	for row in csv_reader:
		input_x_test.append([float(x)/255 for x in row[:len(row)-1]])
		input_y_test.append(float(row[len(row)-1]))
	datax_test = np.array(input_x_test)
	datax_test_temp = np.array(input_x_test)
	datay_test = np.array(input_y_test)
	datay_test_temp = np.array(input_y_test)
	number_of_test = datax_test.shape[0]
	result1 = clf1.predict(datax_test)
	correct_count = 0.0
	correct_count1 = 0.0
	total_count = 0.0
	for i in range(0,number_of_test):
		total_count += 1
		if(result1[i]==datay_test[i]):
			correct_count1 +=1
	print("The accuracy from the given data by gausian kernal methord  comes out to be "+ str(100*float(correct_count1/total_count))+"%")

end_testing_time = time.time()-start_time
print("The testing time is "+str(end_testing_time))