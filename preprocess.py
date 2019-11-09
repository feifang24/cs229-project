import os
import random

'''
Make sure raw data is aligned like mentioned in SPEC.txt, and just run
$ python preprocessor.py
to generate data directory.
'''

def load_data(path):
  '''
  Returns a list of 3-tuple <label, score, text> of all pos/neg examples
  '''
  allData = []
  for label in ["neg", "pos"]:
    trainingDataDir = os.path.join(path, label)
    for filename in os.listdir(trainingDataDir):
      if not filename.endswith("txt"):
        continue
      score = filename.split("_")[-1][0]
      with open(os.path.join(trainingDataDir, filename), "r") as f:
        text = f.read().strip().replace("<br />", " ")
        allData.append((label, score, text))
  return allData


def sample(randomData, k):
  return list(randomData)[:k]


def write_files(outputDir, data):
  if not os.path.exists(outputDir):
      os.makedirs(outputDir)
  for i,example in enumerate(data):
    outputFilename = "_".join([str(i), example[0], example[1]]) + ".txt"
    with open(os.path.join(outputDir, outputFilename), "w+") as f:
        f.write(example[2])


def main():
  # Output dir names
  outputDataDir = "data"
  testDataDir = "test"
  ogDataDir = "og"
  smallDataDir = "sd"
  # Input dir names
  rawDataDir = "imdb"
  trainFolder = "train"
  testFolder = "test"

  '''
  if os.path.exists(outputDataDir):
    print('\"%s\" already exists as a directory. Delete it before regenerating data.' % outputDataDir)
    return
  '''

  trainingDataPath = os.path.join(rawDataDir, trainFolder)
  allTrainingData = load_data(trainingDataPath)
  random.seed(42)
  random.shuffle(allTrainingData)
  # write sampled filesets
  for k in [100,200,400,800,1600,3200]:
    subset = sample(allTrainingData, k)
    sdOutputDir = os.path.join(outputDataDir, smallDataDir+str(k))
    if os.path.exists(outputDataDir):
      continue
    write_files(sdOutputDir, subset)

  # write og fileset
  ogOutputDir = os.path.join(outputDataDir, ogDataDir)
  write_files(ogOutputDir, allTrainingData)

  # write test fileset
  testDataPath = os.path.join(rawDataDir, testFolder)
  allTestData = load_data(testDataPath)
  testOutputDir = os.path.join(outputDataDir, testDataDir)
  write_files(testOutputDir, allTestData)


main()
