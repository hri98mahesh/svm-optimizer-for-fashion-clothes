import csv
import math
import nltk
from random import seed
from random import randint
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import re
import string


def convert(line,stop_words,punc):
	line = line.lower()
	s = list(line)
	word_tokens = ''.join([o for o in s if not o in punc]).split()
	ret =  [w for w in word_tokens if w not in stop_words and not re.match("@.*",w)] 
	ret1 = list(nltk.bigrams(ret))
	ret2 = list(nltk.ngrams(ret,3))
	ret3 = list(nltk.ngrams(ret,4))
	return ret,ret1,ret2,ret3

def update_dict(Dict,lst):
	for i in lst:
		if(i in Dict):
			Dict.update({i:Dict[i]+1})
		else:
			Dict.update({i:1})
	return Dict

def convert_prob(probablity):
	tmp_probablity= [0.0,0.0]
	if probablity[0]<=probablity[1]:
		tmp_probablity[0] = math.exp(0)/(math.exp(0)+math.exp(probablity[1]-probablity[0]))
		tmp_probablity[1] = 1-tmp_probablity[0]
	else:
		tmp_probablity[1] = math.exp(0)/(math.exp(0)+math.exp(probablity[0]-probablity[1]))
		tmp_probablity[0] = 1-tmp_probablity[1]
	return tmp_probablity

def predict1(th,probablity):
	probablity = convert_prob(probablity)
	if(th*(probablity[0]+probablity[1]) <= probablity[0]):
		return 0
	else: 
		return 1

def predict_random():
	return randint(0,1)

def predict_data(th,Dict0,Dict1,Dict,lst,phi_y,Dict0_bigram,Dict1_bigram,Dict_bigram,lst1,phi_y_bi,Dict0_trigram,Dict1_trigram,Dict_trigram,lst2,phi_y_tri,Dict0_quadgram,Dict1_quadgram,Dict_quadgram,lst3,phi_y_quad):
	tmp_probablity = [0.0,0.0]
	for i in lst:
		if i in Dict0:
			tmp_probablity[0] += math.log(Dict0[i]+1) - math.log(phi_y[0]+len(Dict))
		else:
			tmp_probablity[0] += math.log(1) - math.log(phi_y[0]+len(Dict))
		if i in Dict1:
			tmp_probablity[1] += math.log(Dict1[i]+1) - math.log(phi_y[1]+len(Dict))
		else:
			tmp_probablity[1] += math.log(1) - math.log(phi_y[1]+len(Dict))
	for i in lst1:
		if i in Dict0_bigram :
			tmp_probablity[0] += math.log(Dict0_bigram[i]+1) - math.log(phi_y_bi[0]+len(Dict_bigram))
		else:
			tmp_probablity[0] += math.log(1) - math.log(phi_y_bi[0]+len(Dict_bigram))
		if i in Dict1_bigram :
			tmp_probablity[1] += math.log(Dict1_bigram[i]+1) - math.log(phi_y_bi[1]+len(Dict_bigram))
		else:
			tmp_probablity[1] += math.log(1) - math.log(phi_y_bi[1]+len(Dict_bigram))
	for i in lst2:
		if i in Dict0_trigram :
			tmp_probablity[0] += math.log(Dict0_trigram[i]+1) - math.log(phi_y_tri[0]+len(Dict_trigram))
		else:
			tmp_probablity[0] += math.log(1) - math.log(phi_y_tri[0]+len(Dict_trigram))
		if i in Dict1_trigram :
			tmp_probablity[1] += math.log(Dict1_trigram[i]+1) - math.log(phi_y_tri[1]+len(Dict_trigram))
		else:
			tmp_probablity[1] += math.log(1) - math.log(phi_y_tri[1]+len(Dict_trigram))
	for i in lst3:
		if i in Dict0_quadgram :
			tmp_probablity[0] += math.log(Dict0_quadgram[i]+1) - math.log(phi_y_quad[0]+len(Dict_quadgram))
		else:
			tmp_probablity[0] += math.log(1) - math.log(phi_y_quad[0]+len(Dict_quadgram))
		if i in Dict1_quadgram :
			tmp_probablity[1] += math.log(Dict1_quadgram[i]+1) - math.log(phi_y_quad[1]+len(Dict_quadgram))
		else:
			tmp_probablity[1] += math.log(1) - math.log(phi_y_quad[1]+len(Dict_quadgram))
	return predict1(th,tmp_probablity),tmp_probablity

def print1(Dict,phi):
	print(Dict)
	print(phi)

# stop_words = set(stopwords.words('english')) 
stop_words ={"shan't", 'only', 'before', 'himself', 'over', 'he', 'can', 'an', 'this', 'into', 'if', 'shan', 'yours', 'doing', 'about', 'all', "aren't", 'below', 'having', "you're", 'any', "shouldn't", "didn't", 'it', 'to', 'on', "you'll", 'hadn', "wasn't", 'and', "hadn't", 'off', 'you', 'until', 'themselves', 'will', "should've", 'haven', 'then', 'hasn', 'doesn', "isn't", 'after', 'their', 'ma', 'here', 'll', "you'd", 'yourselves', 'does', 'is', 'as', 'there', 'm', 'his', 'down', 'now', 'very', 'for', 'y', 'who', 'am', 'be', 'shouldn', 'few', 'these', 'once', 'nor', 'were', 'wouldn', 'has', 'so', "hasn't", 'that', 'out', 's', 'the', 'him', 'your', 'in', 'where', 'from', 'under', 'through', 're', 'weren', "doesn't", 'our', 'other', "she's", 'o', "haven't", 'why', 'd', "don't", 'between', 'won', 'both', 't', 'was', 'aren', 'up', 'i', "mustn't", 'above', "mightn't", 'being', 'ours', 'mightn', 'mustn', 'than', 'too', 'whom', 'by', "wouldn't", 'during', 'a', 'of', 'did', 'she', 'had', 'herself', 'or', 'not', 'should', 'again', 'each', 'wasn', "that'll", 'we', 'isn', 'own', 'but', 'have', 'such', "couldn't", 'more', 'theirs', 'no', 'itself', 'most', 'do', "it's", 'which', 'how', 'ain', 'because', 'they', 'hers', 'its', "needn't", 'with', 'been', 'her', "weren't", 'my', 'needn', 'at', 'when', 'some', 'don', 'are', 'them', 'couldn', 'me', "you've", 'didn', 'those', 'what', 'further', 'ourselves', 'same', "won't", 'while', 'yourself', 've', 'myself', 'against', 'just'}
ps = PorterStemmer() 
# punc = string.punctuation
pattern = re.compile("@.*")
punc = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
tknzr = TweetTokenizer(strip_handles=True)
Dict0 = {}
Dict1 = {}
Dict = {}
phi_y = [0.0,0.0]
c =0;
Dict0_bigram = {}
Dict1_bigram = {}
Dict0_trigram = {}
Dict1_trigram = {}
Dict0_quadgram = {}
Dict1_quadgram = {}
with open('data/train.csv',encoding='latin-1') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter = ',')
	line_count = 0
	for row in csv_reader:
		c = c+1
		if c%100000==0:
			print(c)
		lst,lst1,lst2,lst3 = convert(row[5],stop_words,punc)
		Dict = update_dict(Dict,lst)
		if row[0] =='0':
			phi_y[0] = phi_y[0]+len(lst)
			Dict0 = update_dict(Dict0,lst)
			Dict0_bigram = update_dict(Dict0_bigram,lst1)
			Dict0_trigram = update_dict(Dict0_trigram,lst2)
			Dict0_quadgram = update_dict(Dict0_quadgram,lst3)
		if row[0] =='4':
			phi_y[1] = phi_y[1]+len(lst)
			Dict1 = update_dict(Dict1,lst)
			Dict1_bigram = update_dict(Dict1_bigram,lst1)
			Dict1_trigram = update_dict(Dict1_trigram,lst2)
			Dict1_quadgram = update_dict(Dict1_quadgram,lst3)
Dict0_bigramT = {}
Dict1_bigramT = {}
Dict_bigram = {}
phi_y_bi = [0.0,0.0]
threshold_bi = 12
Dict0_trigramT = {}
Dict1_trigramT = {}
Dict_trigram = {}
phi_y_tri = [0.0,0.0]
threshold_tri = 9
Dict0_quadgramT = {}
Dict1_quadgramT = {}
Dict_quadgram = {}
phi_y_quad = [0.0,0.0]
threshold_quad = 7
for key in  Dict0_bigram.keys():
	if Dict0_bigram[key] >= threshold_bi:
		Dict0_bigramT.update({key:Dict0_bigram[key]})
		phi_y_bi[0] += Dict0_bigram[key]
		Dict_bigram.update({key:Dict0_bigram[key]})

for key in  Dict1_bigram.keys():
	if Dict1_bigram[key] >= threshold_bi:
		Dict1_bigramT.update({key:Dict1_bigram[key]})
		phi_y_bi[1] += Dict1_bigram[key]
		Dict_bigram.update({key:Dict1_bigram[key]})

for key in  Dict0_trigram.keys():
	if Dict0_trigram[key] >= threshold_tri:
		Dict0_trigramT.update({key:Dict0_trigram[key]})
		phi_y_tri[0] += Dict0_trigram[key]
		Dict_trigram.update({key:Dict0_trigram[key]})

for key in  Dict1_trigram.keys():
	if Dict1_trigram[key] >= threshold_tri:
		Dict1_trigramT.update({key:Dict1_trigram[key]})
		phi_y_tri[1] += Dict1_trigram[key]
		Dict_trigram.update({key:Dict1_trigram[key]})

for key in  Dict0_quadgram.keys():
	if Dict0_quadgram[key] >= threshold_quad:
		Dict0_quadgramT.update({key:Dict0_quadgram[key]})
		phi_y_quad[0] += Dict0_quadgram[key]
		Dict_quadgram.update({key:Dict0_quadgram[key]})

for key in  Dict1_quadgram.keys():
	if Dict1_quadgram[key] >= threshold_quad:
		Dict1_quadgramT.update({key:Dict1_quadgram[key]})
		phi_y_quad[1] += Dict1_quadgram[key]
		Dict_quadgram.update({key:Dict1_quadgram[key]})

with open('data/test.csv',encoding='latin-1') as csv_file:
	confusion_matrix = [[0.0,0.0],[0.0,0.0]]
	csv_reader = csv.reader(csv_file,delimiter = ',')
	correct_count = 0.0
	correct_count_majority =0.0
	correct_count_random = 0.0
	total_count = 0.0
	c1 = 0
	test_x = []
	test_y = []
	for row in csv_reader:
		if(int(row[0])!=2):
			test_x.append(row[5])
			test_y.append(row[0])
			lst,lst1,lst2,lst3 = convert(row[5],stop_words,punc)
			lst1 = [i for i in lst1 if i in Dict_bigram]
			lst2 = [i for i in lst2 if i in Dict_trigram]
			lst3 = [i for i in lst3 if i in Dict_quadgram]
			predict,temp = predict_data(0.5,Dict0,Dict1,Dict,lst,phi_y,Dict0_bigramT,Dict1_bigramT,Dict_bigram,lst1,phi_y_bi,Dict0_trigramT,Dict1_trigramT,Dict_trigram,lst2,phi_y_tri,Dict0_quadgramT,Dict1_quadgramT,Dict_quadgram,lst3,phi_y_quad)
			# predict_maj = predict1(0.5,phi_y)
			# predict_rand = predict_random()
			if int(row[0])==0:
				c1 +=1
			if 4*predict == int(row[0]):
				if(predict==1):
					confusion_matrix[1][1]+=1
				else:
					confusion_matrix[0][0]+=1
				correct_count +=1
			else:
				if(predict==1):
					confusion_matrix[0][1]+=1
				else:
					confusion_matrix[1][0]+=1
			# if 4*predict_maj == int(row[0]):
			# 	correct_count_majority +=1
			# if 4*predict_rand == int(row[0]):
			# 	correct_count_random +=1
			total_count += 1

accuracy = float(correct_count/total_count)
# accuracy_maj = float(correct_count_majority/total_count)
# accuracy_rand = float(correct_count_random/total_count)
print("the accuracy from Training is "+str(accuracy))
print(confusion_matrix)

def roc(th,stop_words,punc,test_x,test_y,Dict0,Dict1,Dict,phi_y,Dict0_bigramT,Dict1_bigramT,Dict_bigram,phi_y_bi,Dict0_trigramT,Dict1_trigramT,Dict_trigram,phi_y_tri,Dict0_quadgramT,Dict1_quadgramT,Dict_quadgram,lst3,phi_y_quad):
	true_pos = 0.0
	false_pos = 0.0
	true_neg = 0.0
	false_neg = 0.0
	for i in range(0,len(test_x)):
		lst,lst1,lst2,lst3 = convert(test_x[i],stop_words,punc)
		lst1 = [i for i in lst1 if i in Dict_bigram]
		lst2 = [i for i in lst2 if i in Dict_trigram]
		lst3 = [i for i in lst3 if i in Dict_quadgram]
		predict,temp = predict_data(th,Dict0,Dict1,Dict,lst,phi_y,Dict0_bigramT,Dict1_bigramT,Dict_bigram,lst1,phi_y_bi,Dict0_trigramT,Dict1_trigramT,Dict_trigram,lst2,phi_y_tri,Dict0_quadgramT,Dict1_quadgramT,Dict_quadgram,lst3,phi_y_quad)
		if int(test_y[i])==0:
			if(predict==0):
				true_pos += 1.0
			else:
				false_neg += 1.0
		else:
			if(predict==1):
				true_neg += 1.0
			else:
				false_pos += 1.0
	print(str(true_pos)+ " "+str(false_pos))
	tpr_rate = true_pos/(true_pos+false_neg)
	fpr_rate = false_pos/(true_neg+false_pos)
	return (tpr_rate,fpr_rate)


x_axis = []
y_axis = []
for i in range(0,100):
	(tpr,fpr) = roc(i*0.01,stop_words,punc,test_x,test_y,Dict0,Dict1,Dict,phi_y,Dict0_bigramT,Dict1_bigramT,Dict_bigram,phi_y_bi,Dict0_trigramT,Dict1_trigramT,Dict_trigram,phi_y_tri,Dict0_quadgramT,Dict1_quadgramT,Dict_quadgram,lst3,phi_y_quad)
	y_axis.append(tpr)
	x_axis.append(fpr)

print(y_axis)
print(x_axis)
plt.plot(x_axis,y_axis)
plt.ylabel('FPR')
plt.xlabel('PR')
plt.show()
plt.savefig('roc_image')



