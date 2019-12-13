from dataprocessor import ImdbProcessor
import collections
import numpy as np
from tokenizer import BasicTokenizer,FullTokenizer


def labels_to_vector(labels):
  '''
  Converts a list of "pos"/"neg" labels to a numpy array of 1 and 0

  Return:
    (n) array representation of examples
  '''
  return np.array([1 * (label == "pos") for label in labels])

def reg_labels_to_vector(labels):
  return np.asarray(labels)


def example_to_indices(example, tokenizer, dictionary):
  '''
  Converts a common.InputExample to a list of indices

  Pad short example with 1, and replace unrecognized words with 0
  '''
  tokens = tokenizer.tokenize(example.text_a)
  return [dictionary.get(token, 0) for token in tokens]


def examples_to_list_of_indices(examples, tokenizer, dictionary):
  '''
  Converts a list of common.InputExample to a list of indices

  Args:
    examples: list of common.InputExample (size n)

  Return:
    A list (of length n) of lists (of variable sizes) representing the indices of tokens in this example
  '''
  return [example_to_indices(example, tokenizer, dictionary) for example in examples]


def construct_dictionary(examples, tokenizer, K, removeTopWords):
  '''
  Construct dictionaries for our corpus. 

  Index 0 is reserved for unspecified words.

  Args:
    examples: a list of common.InputExample
    K: only construct index for top K most frequent words

  Return:
    dictionary: maps token to index
    reverseDictionary: maps index to token
  '''
  rawCounts = collections.Counter()
  for example in examples:
    tokens = tokenizer.tokenize(example.text_a)
    rawCounts.update(tokens)
  mostCommonTokens = rawCounts.most_common(K)[removeTopWords:]
  dictionary = dict()
  reverseDictionary = [None]
  index = 1
  for token,_ in mostCommonTokens:
    reverseDictionary.append(token)
    dictionary[token] = index
    index += 1
  return dictionary, reverseDictionary


def test():
  tokenizer = FullTokenizer("vocab.txt")
  print(tokenizer.tokenize("eat"))