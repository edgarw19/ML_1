from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

i = 0
targets = []
with open("out_classes_5.txt") as f:
	for line in f:
		targets.append(int(line))

bag = []
with open("out_bag_of_words_5.csv") as f:
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
print(test_targets)

test_bag = []
with open("test_bag_of_words_0.csv") as f:
	for line in f:
		splitLine = line.split(",")
		one_bag = []
		for x in splitLine:
			one_bag.append(int(x))
		test_bag.append(one_bag)

clf = MultinomialNB()
clf.fit(bag, targets)
predicted = clf.predict(bag)
target_names = ["Positive", "Negative"]
print("RESULTS ON TRAINING DATA")
print(metrics.classification_report(targets, predicted, target_names=target_names))
print()
print("RESULTS ON TEST DATA")

test_predicted = clf.predict(test_bag)
print(metrics.classification_report(test_targets, test_predicted, target_names=target_names))
