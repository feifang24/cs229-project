from dataprocessor import ImdbProcessor
import collections
import numpy as np

def construct_imdb_dictionaries():
  '''
  Return:
    dictionary: maps word to index
    reverseDictionary: maps index to word
  '''
  dataDir = '../imdb-data'
  dataProcessor = ImdbProcessor(dataDir)
  trainExample, devExample = dataProcessor.get_train_and_dev_examples('og')
  allExample = trainExample + devExample
  dictionary, reverseDictionary = construct_dictionaries(allExample)
  return dictionary


def examples_to_matrix(examples, dictionary, maxlen=80):
  '''
  Converts a list of common.InputExample to a numpy matrix representation of data

  Args:
    examples: list of common.InputExample (size n)
    maxlen: the maximum of each vector

  Return:
    (n x maxlen) array representation of examples
  '''
  n = len(examples)
  arr = np.zeros((n, maxlen))
  for i,example in enumerate(examples):
    arr[i] = example_to_vector(example, dictionary, maxlen)
  return arr


def labels_to_vector(labels):
  '''
  Converts a list of "pos"/"neg" labels to a numpy array of 1 and 0

  Return:
    (n) array representation of examples
  '''
  return np.array([1 * (label == "pos") for label in labels])


def example_to_vector(example, dictionary, maxlen=80):
  '''
  Converts a common.InputExample to a numpy array of length maxlen

  Pad short example with 1, and replace unrecognized words with 0
  '''
  vec = np.ones(maxlen)
  for i,word in enumerate(text_to_tokens(example.text_a)):
    if i >= maxlen:
      break
    if word not in dictionary:
      vec[i] = 0
    else:
      vec[i] = dictionary[word]
  return vec


def construct_dictionaries(examples, K=10000):
  '''
  Construct dictionaries for our corpus. 

  Index 0 is reserved for unspecified words. Index 1 is reserved for padding.

  Args:
    examples: a list of common.InputExample
    K: only construct index for top K most frequent words

  Return:
    dictionary: maps word to index
    reverseDictionary: maps index to word
  '''
  rawCounts = collections.Counter()
  for example in examples:
    words = text_to_tokens(example.text_a)
    rawCounts.update(words)
  mostCommonWords = rawCounts.most_common(K)
  dictionary = dict()
  reverseDictionary = [None, None]
  index = 2
  for word,_ in mostCommonWords:
    reverseDictionary.append(word)
    dictionary[word] = index
    index += 1
  return dictionary, reverseDictionary


def text_to_tokens(text):
  return [word.lower() for word in text.split()]

