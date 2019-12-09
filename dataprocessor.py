from common import InputExample
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv

class ImdbProcessor():

  def __init__(self, dataDirPath):
    '''
    dataDirPath is the path that leads to "data" directory that preprocessor outputted.
    '''
    self.dataDirPath = dataDirPath
    self.allDataSets = set(["og", "wd01", "sd100", "sd200", "sd400", "sd800", 
                            "sd1600", "sd3200", "sd6400", "sd12800", "dev", "test"])

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, dataset):
    if dataset.startswith("wd"):
      # load snorkel dataset from csv file
      return self._create_examples_from_csv(os.path.join(self.dataDirPath, "{}.csv".format(dataset)))
    return self._create_examples(os.path.join(self.dataDirPath, dataset))

  def get_dev_examples(self):
    return self._create_examples(os.path.join(self.dataDirPath, "dev"))

  def get_test_examples(self, splitLabel=False):
    testExamples = self._create_examples(os.path.join(self.dataDirPath, "test"))
    if not splitLabel:
      return testExamples
    posExamples = []
    negExamples = []
    for example in testExamples:
      if example.label == "pos":
        posExamples.append(example)
      else:
        negExamples.append(example)
    return posExamples, negExamples

  def _create_examples(self, dataDirPath):
    examples = []
    for filename in tf.gfile.ListDirectory(dataDirPath):
      if not filename.endswith("txt"):
        continue
      keys = filename.split(".")[0].split("_")
      assert len(keys) == 3
      # keys is [id, label, review_score]. For now we are only interested in the label
      label = keys[1]
      with tf.gfile.Open(os.path.join(dataDirPath, filename)) as f:
        text = f.read().strip().replace("<br />", " ")
      examples.append(InputExample(
          guid="unused_id", text_a=text, text_b=None, label=label))
    return examples

  def _create_examples_from_csv(self, input_file):
    """Reads a comma separated value file."""
    examples = []
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f)
      for line in reader:
        text, label = line
        text = text.strip().replace("<br />", " ")
        str_label = 'pos' if label == '1' else 'neg'
        examples.append(InputExample(
          guid="unused_id", text_a=text, text_b=None, label=str_label))
    return examples

