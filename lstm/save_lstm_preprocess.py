import collections
import numpy as np
from tokenizer import BasicTokenizer,FullTokenizer

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from dataprocessor import ImdbProcessor

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


def examples_to_matrix_2(examples, tokenizer, maxlen=80):
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
    arr[i] = example_to_vector_2(example, tokenizer, maxlen)
  return arr


def labels_to_vector(labels):
  '''
  Converts a list of "pos"/"neg" labels to a numpy array of 1 and 0

  Return:
    (n) array representation of examples
  '''
  return np.array([1 * (label == "pos") for label in labels])


def example_to_vector(example, dictionary, maxlen):
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


def example_to_vector_2(example, tokenizer, maxlen):
  '''
  Converts a common.InputExample to a numpy array of length maxlen

  Pad short example with 1, and replace unrecognized words with 0
  '''
  vec = np.ones(maxlen)
  tokens = tokenizer.tokenize(example.text_a)
  ids = tokenizer.convert_tokens_to_ids(tokens)
  if len(ids) >= maxlen:
    return ids[:maxlen]
  else:
    ids = ids + [1] * (maxlen - len(ids))
  return ids


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


def construct_dictionaries_3(examples, tokenizer, K, removeTopWords):
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
    words = tokenizer.tokenize(example.text_a)
    rawCounts.update(words)
  mostCommonWords = rawCounts.most_common(K)[removeTopWords:]
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


def construct_vocabs(examples, K=10000):
  '''
  Construct dictionaries for our corpus. 

  Args:
    examples: a list of common.InputExample
    K: only construct index for top K most frequent words

  Return:
    dictionary: maps word to index
    reverseDictionary: maps index to word
  '''
  rawCounts = collections.Counter()
  for example in examples:
    words = [word.lower() for word in example.text_a.split()]
    rawCounts.update(words)
  mostCommonWords = rawCounts.most_common(K)
  return ['[UNK]', 'SomeGibberishThatStandsForPaddingThatNoOneShouldUse'] + [pair[0] for pair in mostCommonWords]


def test():
  tokenizer = FullTokenizer("vocab.txt")
  print(tokenizer.tokenize("eat"))

test()