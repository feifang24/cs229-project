from common import InputExample
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

class ImdbProcessor():

  def __init__(self, dataDirPath):
    '''
    dataDirPath is the path that leads to "data" directory that preprocessor outputted.
    '''
    self.dataDirPath = dataDirPath
    self.allDataSets = set(["og", "sd100", "sd200", "sd400", "sd800", "sd1600", "sd3200", "sd6400", "sd12800", "test"])
        
    self.trainDevSplit = {ds: 0.8 for ds in self.allDataSets}
    self.trainDevExamples = None

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, dataset):
    trainExamples, _ = self._get_train_dev_examples(dataset)
    return trainExamples

  def get_dev_examples(self, dataset):
    _, devExamples = self._get_train_dev_examples(dataset)
    return devExamples

  def get_train_and_dev_examples(self, dataset):
    return self._get_train_dev_examples(dataset)

  def get_test_examples(self):
    return self._create_examples(os.path.join(self.dataDirPath, "test"))

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

  def _get_train_dev_examples(self, dataset):
    assert dataset in self.allDataSets
    if self.trainDevExamples is None:
      # load examples if they haven't been loaded before
      self.trainDevExamples = self._create_examples(os.path.join(self.dataDirPath, dataset))
    return train_test_split(self.trainDevExamples, 
                        train_size=self.trainDevSplit[dataset], random_state=42)
