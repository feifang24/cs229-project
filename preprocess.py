import os
import random

'''
Make sure raw data is aligned like mentioned in SPEC.txt, and just run
$ python preprocessor.py
to generate data directory.
'''

SAMPLE_DS_SIZES = [100,200,400,800,1600,3200]

def load_data(path):
  '''
  Returns a list of 3-tuple <label, score, text> of all pos/neg examples
  '''
  allLabels = ["neg", "pos"]
  allData = {label: [] for label in allLabels}
  for label in allLabels:
    trainingDataDir = os.path.join(path, label)
    for filename in os.listdir(trainingDataDir):
      if not filename.endswith("txt"):
        continue
      score = filename.split("_")[-1][0]
      with open(os.path.join(trainingDataDir, filename), "r") as f:
        text = f.read().strip().replace("<br />", " ")
        allData[label].append((label, score, text))
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
  outputDataDir = "imdb-data"
  testDataDir = "test"
  ogDataDir = "og"
  smallDataDir = "sd"
  # Input dir names
  rawDataDir = "imdb"
  trainFolder = "train"
  testFolder = "test"


  trainingDataPath = os.path.join(rawDataDir, trainFolder)
  allTrainingData = load_data(trainingDataPath)
  posTrainingData = allTrainingData['pos']
  negTrainingData = allTrainingData['neg']
  random.seed(42)
  random.shuffle(posTrainingData)
  random.shuffle(negTrainingData)

  # write sampled filesets
  for k in SAMPLE_DS_SIZES:
    sdOutputDir = os.path.join(outputDataDir, smallDataDir+str(k))
    if os.path.exists(sdOutputDir):
      continue
    subset = sample(posTrainingData, int(k/2)) + sample(negTrainingData, int(k/2))
    write_files(sdOutputDir, subset)

  # write og fileset
  ogOutputDir = os.path.join(outputDataDir, ogDataDir)
  if not os.path.exists(ogOutputDir):
    write_files(ogOutputDir, posTrainingData + negTrainingData)

  # write test fileset
  testDataPath = os.path.join(rawDataDir, testFolder)
  testOutputDir = os.path.join(outputDataDir, testDataDir)
  if not os.path.exists(testOutputDir):
    allTestData = load_data(testDataPath)
    write_files(testOutputDir, allTestData)


main()
