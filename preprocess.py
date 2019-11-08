import os
import random

rawDataDir = "imdb"
outputDataDir = "data"
ogDataDir = "og"
smallDataDir = "sd"
trainFolder = "train"
testFolder = "test"

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
  trainingDataPath = os.path.join(rawDataDir, trainFolder)
  allTrainingData = load_data(trainingDataPath)
  random.shuffle(allTrainingData)
  for k in [100,200,400,800,1600,3200]:
    subset = sample(allTrainingData, k)
    outputDir = os.path.join(outputDataDir, smallDataDir+str(k), trainFolder)
    write_files(outputDir, subset)
  outputDir = os.path.join(outputDataDir, ogDataDir, trainFolder)
  write_files(outputDir, allTrainingData)


main()
