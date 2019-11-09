from common import InputExample
import os

class ImdbProcessor():

  def __init__(self, dataDirPath):
    self.dataDirPath = dataDirPath
    self.allDataSets = set(["og", "sd100", "sd200", "sd400", "sd800", "sd1600", "sd3200", "test"])

  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, dataset):
    assert dataset in self.allDataSets
    return self._create_examples(os.path.join(self.dataDirPath, dataset))

  def get_dev_examples(self):
    # TODO: What do we do here? Is dev set required?
    raise NotImplementedError()

  def get_test_examples(self):
    return self._create_examples(os.path.join(self.dataDirPath, "test"))

  def _create_examples(self, dataDirPath):
    examples = []
    for filename in os.listdir(dataDirPath):
      if not filename.endswith("txt"):
        continue
      keys = filename.split(".")[0].split("_")
      assert len(keys) == 3
      # keys is [id, label, review_score]. For now we are only interested in the label
      label = keys[1]
      with open(os.path.join(dataDirPath, filename)) as f:
        text = f.read().strip().replace("<br />", " ")
      examples.append(InputExample(
          guid="unused_id", text_a=text, text_b=None, label=label))
    return examples