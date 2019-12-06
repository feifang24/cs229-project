import secrets
import os

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


def random_model_hash():
  return secrets.token_hex(nbytes=4)


def write_files(outputDir, data):
  if not os.path.exists(outputDir):
      os.makedirs(outputDir)
  for i,example in enumerate(data):
    outputFilename = "_".join([str(i), example[0], example[1]]) + ".txt"
    with open(os.path.join(outputDir, outputFilename), "w+") as f:
        f.write(example[2])