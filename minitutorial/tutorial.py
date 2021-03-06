from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy
import re
import sys
import getopt
import codecs
import time
import os
import csv

def main(argv):
	# print("HERE")

	start_time = time.time()

	path = ''
	outputf = 'out'
	vocabf = ''
	data = ''
	wordThreshold = 3
	bigramCount = 3

	try:
		opts, args = getopt.getopt(argv,"p:t:o:v:c:b:",["path=","data=", "ofile=","vocabfile=", "count=", "bigrams="])
	except getopt.GetoptError:
		print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
		  print 'Usage: \n python preprocessSentences.py -p <path> -o <outputfile> -v <vocabulary>'
		  sys.exit()
		elif opt in ("-p", "--path"):
		  path = arg
		elif opt in ("-o", "--ofile"):
		  outputf = arg
		elif opt in ("-v", "--vocabfile"):
		  vocabf = arg
		elif opt in ("-t", "--data"):
		  # print("CAME HERE")
		  data = arg
		  # print(opt, arg)
		elif opt in ("-c", "--count"):
		  wordThreshold = int(arg)
		elif opt in ("-b", "--bigramcount"):
		  bigramCount = int(arg)
	print(opts)
	i = 0
	targets = []
	# print("WORD THRESHOLD", wordThreshold)

	with open("out_classes_" + str(wordThreshold) + ".txt") as f:
		for line in f:
			targets.append(int(line))

	sentences = []
	with open("train.txt") as f:
		for line in f:
			sentences.append(line)

	bag = []
	with open("out_bag_of_words_" + str(wordThreshold) + ".csv") as f:
		for line in f:
			splitLine = line.split(",")
			one_bag = []
			for x in splitLine:
				one_bag.append(int(x))
			bag.append(one_bag)

	test_targets = []
	with open("test_classes_0.txt") as f:
		for line in f:
			test_targets.append(int(line))

	test_bag = []
	with open("test_bag_of_words_0.csv") as f:
		for line in f:
			splitLine = line.split(",")
			one_bag = []
			for x in splitLine:
				one_bag.append(int(x))
			test_bag.append(one_bag)

	target_names = ["Positive", "Negative"]
	X_new = SelectKBest(score_func=chi2, k=710).fit(bag, targets)
	clf = MultinomialNB()
	clf.fit(X_new.transform(bag), targets)
	predicted = clf.predict(X_new.transform(bag))
	for i in range(0, len(predicted)):
		if (predicted[i] != targets[i]):
			print(sentences[i])

	
	# print()
	
	test_predicted = clf.predict(X_new.transform(test_bag))

	
	# for i in range(0, len(test_predicted)):
	# 	if (test_predicted[i] != test_targets[i]):
	# 		print(sentences[i])
	print("RESULTS ON TRAINING DATA")
	print(metrics.classification_report(targets, predicted, target_names=target_names, digits=4))
	print("RESULTS ON TEST DATA")
	print(metrics.classification_report(test_targets, test_predicted, target_names=target_names, digits=4))


if __name__ == "__main__":
	main(sys.argv[1:])
