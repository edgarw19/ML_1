import nltk, re, pprint
from nltk import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment import util
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


chars = ['{','}','#','%','&','\(','\)','\[','\]','<','>',',', '.', ';', 
'?', '*', '\\', '\/', '~', '|', '_', '=','+','^',':','\"','\'','@','-']
def numToWords(num,join=True):
    '''words = {} convert an integer number into words'''
    units = ['','one','two','three','four','five','six','seven','eight','nine']
    teens = ['','eleven','twelve','thirteen','fourteen','fifteen','sixteen', \
             'seventeen','eighteen','nineteen']
    tens = ['','ten','twenty','thirty','forty','fifty','sixty','seventy', \
            'eighty','ninety']
    thousands = ['','thousand','million','billion','trillion','quadrillion', \
                 'quintillion','sextillion','septillion','octillion', \
                 'nonillion','decillion','undecillion','duodecillion', \
                 'tredecillion','quattuordecillion','sexdecillion', \
                 'septendecillion','octodecillion','novemdecillion', \
                 'vigintillion']
    words = []
    if num==0: words.append('zero')
    else:
        numStr = '%d'%num
        numStrLen = len(numStr)
        groups = (numStrLen+2)/3
        numStr = numStr.zfill(groups*3)
        for i in range(0,groups*3,3):
            h,t,u = int(numStr[i]),int(numStr[i+1]),int(numStr[i+2])
            g = groups-(i/3+1)
            if h>=1:
                words.append(units[h])
                words.append('hundred')
            if t>1:
                words.append(tens[t])
                if u>=1: words.append(units[u])
            elif t==1:
                if u>=1: words.append(teens[u])
                else: words.append(tens[t])
            else:
                if u>=1: words.append(units[u])
            if (g>=1) and ((h+t+u)>0): words.append(thousands[g]+',')
    if join: return ' '.join(words)
    return words

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
  sentimentAnalyzer = SentimentAnalyzer()
  numberReviews = ["one", "two", "three", "four", "five"]
  numbers = ["1", "2", "3", "4", "5"]
  stopWords = [u'i', u'me', u'my', u'myself', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now', u'd', u'll', u'm', u'o', u're', u've', u'y']
  negatives = [ u'ain', u'aren', u'couldn', u'didn', u'doesn', u'hadn', u'hasn', u'haven', u'isn', u'ma', u'mightn', u'mustn', u'needn', u'shan', u'shouldn', u'wasn', u'weren', u'won', u'wouldn', u'no', u'nor', u'not']
  print(len(stopWords))
  print("ABOVE IS STOPWORDS")
  classes = []
  samples = []
  docs = []
  allWords = []
  sentenceLengths = []
  if train == True:
    words = {}
    nWords = {}
  f = open(path, 'r')
  lines = f.readlines()
  i = 0
  amplification = 2
# ["foo", "bar", "baz"].index("bar")
  for line in lines:
    classes.append(line.rsplit()[-1])
    samples.append(line.rsplit()[0])
    raw = line.decode('latin1')
    # print("RAW 1", raw)
    raw = ' '.join(raw.rsplit()[1:-1])
    raw = re.sub('[%s]' % ''.join(chars), ' ', raw)
    tokens = word_tokenize(raw)
    tokens = [w.lower() for w in tokens]
    tokens = [w for w in tokens if w not in stopWords]
    tokens = [wnl.lemmatize(t) for t in tokens]
    tokens = [porter.stem(t) for t in tokens] 
    # Create dummy array to manipulate
    # Extract everything other than nouns from the dummy array
    # Create a "tokenized array"
    # set tokens equal to array

    # TESTING PART OF SPEECH (WORKED POORLY)
    # dummyTokens = nltk.pos_tag(tokens)
    # tokens = []
    # for i in range(0, len(dummyTokens)):
    #   word, partOfSpeech = dummyTokens[i]
    #   if "JJ" in partOfSpeech:
    #     for i in range(0, amplification+1):
    #       tokens.append(word)
    #   else:
    #     tokens.append(word)

    # get the negation of the word afterward
    for i in range(0, len(tokens)):
      if (tokens[i] in negatives):
        if negatives.index(tokens[i]) > -1 and i < len(tokens) - 2:
          tokens[i+1] = tokens[i] + tokens[i+1]
    tokenBigrams = []
    tokenTrigrams = []

    #Create bigrams
    for i in range(0, len(tokens)):
      if (i < len(tokens)-2):
        phrase = tokens[i] + tokens[i+1]
        tokenBigrams.append(phrase)


    #Create Trigrams
    # for i in range(0, len(tokens)):
    #   if (i < len(tokens) - 3):
    #     phrase = tokens[i] + tokens[i+1] + tokens[i+3]
    #     tokenBigrams.append(phrase)


    allWords = allWords + tokens
    if train == True:
     for t in tokenBigrams:
      try:
        nWords[t] = nWords[t] + 1
      except:
        nWords[t] = 1
     for t in tokens: 
         try:
             text = [t]
             parts = nltk.pos_tag(text)
             word, partOfSpeech = parts[0]
             if "JJ" in partOfSpeech:
              words[t] = words[t] + amplification
             else:
              words[t] = words[t]+1
         except:
             words[t] = 1
     try:
      nWords[("number" + numToWords(len(tokens)))] = nWords[("number" + numToWords(len(tokens)))] + 1
     except:
      nWords[("number" + numToWords(len(tokens)))] = 1
    
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

    vocab = sorted(union(vocab, bigramVocab))
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

 
