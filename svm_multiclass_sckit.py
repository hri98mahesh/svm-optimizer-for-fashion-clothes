import csv
import math
import numpy as np
import time
from cvxopt import matrix
from cvxopt import solvers
from sklearn.svm import SVC

def train(class1,class2,input_data_class1,input_data_class2):
	datax = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datax_temp = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datay = np.ones(datax.shape[0])*(-1)
	datay[:len(input_data_class1)] += 2
	datay_temp = np.ones(datax.shape[0])*(-1)	
	datay_temp[:len(input_data_class1)] += 2
	number_of_training = datax.shape[0]
	number_of_features = datax.shape[1]
	clf = SVC(kernel='linear',gamma='auto')
	clf1 = SVC(kernel='rbf',gamma=0.05)
	clf.fit(datax,datay)
	clf1.fit(datax,datay)
	return clf,clf1

def test(datax_test,input_data_class1,input_data_class2,clf,clf1,class_1,class_2):
	datax = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datax_temp = np.append(np.array(input_data_class1),np.array(input_data_class2),axis=0)
	datay = np.ones(datax.shape[0])*(-1)
	datay[:len(input_data_class1)] += 2
	datay_temp = np.ones(datax.shape[0])*(-1)	
	datay_temp[:len(input_data_class1)] += 2
	datax_test = np.array(datax_test)
	result = clf.predict(datax_test)
	result1 = clf1.predict(datax_test)
	for i in range(0,datax_test.shape[0]):
		if result[i] >=0:
			result[i] = class_1
		else:
			result[i] = class_2
	for i in range(0,datax_test.shape[0]):
		if result1[i] >=0:
			result1[i] = class_1
		else:
			result1[i] = class_2
	return result,result1


start_time = time.time()

with open("fashion_mnist/train.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	input_data = [[],[],[],[],[],[],[],[],[],[]]
	input_total = np.zeros(10)
	for row in csv_reader:
		class_data = float(row[len(row)-1])
		input_data[int(class_data)].append([float(x)/255 for x in row[:len(row)-1]])
		input_total[int(class_data)] += 1
	clf = []
	clf1 = []
	for i  in range(0,10):
		for j in range(i+1,10):
			print(str(i)+" "+str(j))
			temp_clf,temp_clf_1 = train(i,j,input_data[i],input_data[j])
			clf.append(temp_clf)
			clf1.append(temp_clf_1)

end_training_time = time.time()-start_time
print("The Training time is "+ str(end_training_time))

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
	resultK = []
	for i in range(0,10):
		for j in range(i+1,10):
			temp_result,temp_result1 = test(input_x_test,input_data[i],input_data[j],clf[count],clf1[count],i,j)
			result.append(temp_result)
			resultK.append(temp_result1)
			count += 1

resultT = np.transpose(np.array(result))
resultTK = np.transpose(np.array(resultK))
correct_count = 0.0
correct_count1 = 0.0
total_count = 0.0
result_class = []
result_class1 = []

for i in range(0,resultT.shape[0]):
	ch = np.zeros(10)
	for j in range(0,resultT.shape[1]):
		ch[int(resultT[i][j])] += 1
	index = 0 
	max_v = 0
	for j in range(0,10):
		if(ch[j]>max_v):
			index = j
			max_v = ch[j]
	result_class.append(index)
	if(index == input_y_test[i]):
		correct_count += 1
	ch = np.zeros(10)
	for j in range(0,resultTK.shape[1]):
		ch[int(resultTK[i][j])] += 1
	index = 0 
	max_v = 0
	for j in range(0,10):
		if(ch[j]>max_v):
			index = j
			max_v = ch[j]
	result_class1.append(index)
	if(index == input_y_test[i]):
		correct_count1 += 1
	total_count += 1


print(correct_count)
# 4181.0
print(correct_count1)
# 4404.0
print("The total accuracy from linear is "+ str(100*correct_count/total_count)+"%")
# The total accuracy from linear is 83.62%
print("The total accuracy from gausian kernel is "+ str(100*correct_count1/total_count)+"%")
# The total accuracy is 85.06%
print("The total count is "+str(total_count))
confusion_matrix = np.zeros((10,10))
confusion_matrix1 = np.zeros((10,10))
for i in range(0,resultT.shape[0]):
	confusion_matrix[input_y_test[i]][result_class[i]] += 1
	confusion_matrix1[input_y_test[i]][result_class1[i]] += 1
print(confusion_matrix)
print(confusion_matrix1)

# [408.,   2.,  14.,  22.,   0.,   0.,  90.,   0.,   6.,   0.]
# [  3., 483.,   3.,  13.,   2.,   0.,   3.,   0.,   0.,   1.]
# [ 10.,   4., 375.,  11.,  63.,   1.,  66.,   0.,   4.,   0.]
# [ 22.,   9.,  10., 425.,  21.,   0.,  13.,   0.,   0.,   0.]
# [  3.,   0.,  48.,  11., 370.,   0.,  45.,   0.,   3.,   0.]
# [  0.,   0.,   0.,   0.,   0., 457.,   0.,  26.,   3.,  15.]
# [ 47.,   2.,  49.,  16.,  43.,   2., 278.,   0.,  14.,   0.]
# [  0.,   0.,   0.,   0.,   0.,  28.,   0., 459.,   3.,  25.]
# [  7.,   0.,   1.,   2.,   1.,   1.,   5.,   2., 467.,   0.]
# [  0.,   0.,   0.,   0.,   0.,  11.,   0.,  13.,   0., 459.]


# [433.,   1.,   5.,  12.,   3.,   0.,  80.,   0.,   1.,   0.]
# [  0., 482.,   0.,   0.,   1.,   0.,   0.,   0.,   0.,   0.]
# [  5.,   4., 411.,   3.,  41.,   0.,  55.,   0.,   1.,   0.]
# [ 11.,   9.,   7., 457.,  13.,   0.,   9.,   0.,   1.,   0.]
# [  3.,   0.,  37.,   9., 399.,   0.,  34.,   0.,   2.,   0.]
# [  0.,   0.,   0.,   0.,   0., 473.,   0.,  14.,   2.,  11.]
# [ 38.,   4.,  32.,  14.,  38.,   0., 315.,   0.,   2.,   0.]
# [  0.,   0.,   0.,   0.,   0.,  16.,   0., 471.,   2.,  14.]
# [ 10.,   0.,   8.,   5.,   5.,   5.,   7.,   1., 489.,   1.]
# [  0.,   0.,   0.,   0.,   0.,   6.,   0.,  14.,   0., 474.]



