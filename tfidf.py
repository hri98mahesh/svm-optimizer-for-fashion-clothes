import csv
import math
import numpy as np
from random import seed
from random import randint
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


tf_vectorizer = CountVectorizer()
train_X = []
train_Y = []
test_X = []
test_Y = []
with open('data/train.csv',encoding='latin-1') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		
		if int(row[0])==0:
			train_X.append(row[5])
			train_Y.append(0)
		elif int(row[0])==4:
			train_X.append(row[5])
			train_Y.append(1)

with open('data/test.csv',encoding='latin-1') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	for row in csv_reader:
		if int(row[0])==0:
			test_X.append(row[5])
			test_Y.append(0)
		elif int(row[0])==4:
			test_X.append(row[5])
			test_Y.append(1)

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)

X_train_tf = tf_vectorizer.fit_transform(train_X)
X_test_tf = tf_vectorizer.transform(test_X)
naive_bayes_classifier = GaussianNB()
number_of_batches = 1600
number_of_training = train_X.shape[0]
size_of_batch = number_of_training/number_of_batches
for i in range(0,number_of_batches):
	t_X = X_train_tf[int(i*size_of_batch):int((i+1)*size_of_batch)].toarray()
	t_Y = train_Y[int(i*size_of_batch):int((i+1)*size_of_batch)]
	naive_bayes_classifier.partial_fit(t_X,t_Y,classes=np.array([0,1]))
	print(i)
y_pred = naive_bayes_classifier.predict(X_test_tf.toarray())
		
score1 = metrics.accuracy_score(test_Y, y_pred)

print(score1)
# print(metrics.classification_report(test_Y,y_pred,target_names=['Class 2', 'Class 4']))

print(metrics.confusion_matrix(test_Y, y_pred))


