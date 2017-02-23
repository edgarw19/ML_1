import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, isdir, join
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
import numpy
import re
import sys
import getopt
import codecs
import time
import os
import csv


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '!', '.', ';', 
'?', '*', '\\', '\/', '~', '_','|','=','+','^',':','\"','\'','@','-']

def stem(word):
   regexp = r'^(.*?)(ing|ly|ed|ious|ies|ive|es|s|ment)?$'
   stem, suffix = re.findall(regexp, word)[0]
   return stem

def unique(a):
   """ return the list with duplicate elements removed """
   return list(set(a))

def bigrams(a):
  return list(bigrams(a))

def intersect(a, b):
   """ return the intersection of two lists """
   return list(set(a) & set(b))

def union(a, b):
   """ return the union of two lists """
   return list(set(a) | set(b))

def get_files(mypath):
   return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def get_dirs(mypath):
   return [ f for f in listdir(mypath) if isdir(join(mypath,f)) ]

# Reading a bag of words file back into python. The number and order
# of sentences should be the same as in the *samples_class* file.
def read_bagofwords_dat(myfile):
  bagofwords = numpy.genfromtxt('myfile.csv',delimiter=',')
  return bagofwords

def tokenize_corpus(path, train=True):
  # Need to create bigrams in the doc
  # Need to also create a massive doc of all the bigrams
  porter = nltk.PorterStemmer() # also lancaster stemmer
  wnl = nltk.WordNetLemmatizer()
  stopWords = stopwords.words("english")
  print(sort(stopWords))
  print("ABOVE IS STOPWORDS")
  classes = []
  samples = []
  docs = []
  allWords = []
  if train == True:
    words = {}
    nWords = {}
  f = open(path, 'r')
  lines = f.readlines()
  i = 0

  for line in lines:
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = line.decode('latin1')
    raw = ' '.join(raw.rsplit()[1:-1])
    # remove noisy characters; tokenize
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in stopWords]
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [porter.stem(t) for t in tokens] 
    tokenBigrams = []
    for i in range(0, len(tokens)):
      if (i < len(tokens)-2):
        phrase = tokens[i] + tokens[i+1]
        tokenBigrams.append(phrase)


    allWords = allWords + tokens
    if train == True:
     for t in tokenBigrams:
      try:
        nWords[t] = nWords[t] + 1
      except:
        nWords[t] = 1
     for t in tokens: 
         try:
             words[t] = words[t]+1
         except:
             words[t] = 1
    docs.append(tokens)
  # finder = BigramCollocationFinder.from_words(allWords)
  # finder.apply_freq_filter(3)
  # best = finder.nbest(BigramAssocMeasures.pmi, 200)

  if train == True:
     return(docs, classes, samples, words, nWords)
  else:
     return(docs, classes, samples)


def wordcount_filter(words, num=5):
   keepset = []
   for k in words.keys():
       if(words[k] > num):
           keepset.append(k)
   return(sorted(set(keepset)))


def find_wordcounts(docs, vocab):
   bagofwords = numpy.zeros(shape=(len(docs),len(vocab)), dtype=numpy.uint8)
   vocabIndex={}
   for i in range(len(vocab)):
      vocabIndex[vocab[i]]=i

   for i in range(len(docs)):
       doc = docs[i]

       for t in doc:
          index_t=vocabIndex.get(t)
          if index_t>=0:
             bagofwords[i,index_t]=bagofwords[i,index_t]+1

   # print "Finished find_wordcounts for:", len(docs), "docs"
   return(bagofwords)


def main(argv):
  
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
  # print(opts)
  # print("VOCAB" + vocabf)
  # print(wordThreshold)
  traintxt = path+"/" + data + ".txt"
  # print 'Path:', path
  # print 'Training data:', traintxt

  # Tokenize training data (if training vocab doesn't already exist):
  if (not vocabf):
    print(bigramCount)
    # print("WORDCOUNT", wordThreshold)
    word_count_threshold = wordThreshold
    # print(word_count_threshold)
    (docs, classes, samples, words, nWords) = tokenize_corpus(traintxt, train=True)
    vocab = wordcount_filter(words, num=word_count_threshold)
    # print("LEN RIGHT AFTER", len(vocab))
    bigramVocab = wordcount_filter(nWords, bigramCount)

    vocab = union(vocab, bigramVocab)
    print("VOCAB LENGTH", len(vocab))
    # Write new vocab file
    vocabf = outputf+"_vocab_"+str(word_count_threshold)+".txt"
    outfile = codecs.open(path+"/"+vocabf, 'w',"utf-8-sig")
    outfile.write("\n".join(vocab))
    outfile.close()
  else:
    word_count_threshold = 0
    (docs, classes, samples) = tokenize_corpus(traintxt, train=False)
    vocabfile = open(path+"/"+vocabf, 'r')
    vocab = [line.rstrip('\n') for line in vocabfile]
    vocabfile.close()

  # print 'Vocabulary file:', path+"/"+vocabf 

  # Get bag of words:
  bow = find_wordcounts(docs, vocab)
  # Check: sum over docs to check if any zero word counts
  # print "Doc with smallest number of words in vocab has:", min(numpy.sum(bow, axis=1))

  # Write bow file
  with open(path+"/"+outputf+"_bag_of_words_"+str(word_count_threshold)+".csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(bow)

  # Write classes
  outfile= open(path+"/"+outputf+"_classes_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(classes))
  outfile.close()

  # Write samples
  outfile= open(path+"/"+outputf+"_samples_class_"+str(word_count_threshold)+".txt", 'w')
  outfile.write("\n".join(samples))
  outfile.close()

  # print 'Output files:', path+"/"+outputf+"*"

  # # Runtime
  # print 'Runtime:', str(time.time() - start_time)

if __name__ == "__main__":
  main(sys.argv[1:])

 
