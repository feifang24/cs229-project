import tensorflow as tf
import os
from lgf import NaiveLabelGeneratingFunction

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import common

def generate_naive_label(inputPath, lgf):
  '''
  Return a tuple of: (naive_samples, sample_stats)
    naive_samples: a list of (naiveLabel, fakeReviewScore(0), text) pairs
    sample_stats: a dictionary mapping (true_label, naive_label) pairs to counts of occurrences of such pairs
  '''
  sample_stats = {
    ("pos", "pos"):0, ("neg", "pos"):0,
    ("pos", "neg"):0, ("neg", "neg"):0,
    ("pos", None):0, ("neg", None):0
  }
  naive_samples = []
  for filename in tf.gfile.ListDirectory(inputPath):
    if not filename.endswith("txt"):
        continue
    # keys is (id, label, review_score)
    keys = filename.split(".")[0].split("_")
    trueLabel = keys[1]
    assert len(keys) == 3
    with tf.gfile.Open(os.path.join(inputPath, filename)) as f:
        text = f.read().strip().replace("<br />", " ")
    naiveLabel = lgf.label(text)
    if naiveLabel is not None:
      naive_samples.append((naiveLabel, "0", text))
    sample_stats[(trueLabel, naiveLabel)] += 1
  return naive_samples, sample_stats


def format_sample_stats(sample_stats):
  '''
  Format the sample_stats of the naive samples generated by the lgf
  into a human-readable format without writing anything to disk.
  '''
  totalCt = 0
  totalNaiveLabelCt = 0
  posHasNaiveLabelCt = 0
  negHasNaiveLabelCt = 0
  for trueLabel, naiveLabel in sample_stats:
    totalCt += sample_stats[(trueLabel, naiveLabel)]
    if naiveLabel is not None:
      totalNaiveLabelCt += sample_stats[(trueLabel, naiveLabel)]
      if trueLabel == "pos":
        posHasNaiveLabelCt += sample_stats[(trueLabel, naiveLabel)]
      else:
        negHasNaiveLabelCt += sample_stats[(trueLabel, naiveLabel)]
  correctPosLabelCt = sample_stats[("pos", "pos")]
  correctNegLabelCt = sample_stats[("neg", "neg")]
  correctLabelCt = correctPosLabelCt + correctNegLabelCt
  stat = "Total %d files, labeled %d of them, %d of them correct, accuracy %f, recall %f.\n" % (totalCt, totalNaiveLabelCt, correctLabelCt, 1.0*correctLabelCt/totalNaiveLabelCt, 1.0*correctLabelCt/totalCt)
  stat += "Generated %d labels for pos, %d of them correct, accuracy %f, recall %f.\n" % (posHasNaiveLabelCt, correctPosLabelCt, 1.0*correctPosLabelCt/posHasNaiveLabelCt, 2.0*correctPosLabelCt/totalCt)
  stat += "Generated %d labels for neg, %d of them correct, accuracy %f, recall %f.\n" % (negHasNaiveLabelCt, correctNegLabelCt, 1.0*correctNegLabelCt/negHasNaiveLabelCt, 2.0*correctNegLabelCt/totalCt)
  stat += str(sample_stats)
  return stat


def print_naive_sample_stats(inputPath):
  lgf = NaiveLabelGeneratingFunction()
  _, sample_stats = generate_naive_label(inputPath, lgf)
  print(format_sample_stats(sample_stats))


def write_naive_samples_to_data_file(inputPath, outputPath):
  '''
  Write the generated naive samples.
  
  The dataset will be called nwd_xxx, with xxx being the version number here.

  inputPath should probably be equal to outputPath.

  Output:
    1. Will write all training sample in outputPath/ directory, with each training sample
    being a .txt file, and the filename is consistent with the id_label_reviewScore format.
    Since we don't generate reviewScore, it will be a hardcoded number 0. Please don't use it.
    2. Will write a single METADATA text file in outputPath/ directory that records
    the sample_stats of this naive dataset.
  '''
  lgf = NaiveLabelGeneratingFunction()
  directoryNameWithVersion = "nwd00"
  outputDir = outputPath + directoryNameWithVersion

  naive_samples, sample_stats = generate_naive_label(inputPath, lgf)
  formatted_sample_stats = format_sample_stats(sample_stats)
  common.write_files(outputDir, naive_samples)
  with open(outputPath + directoryNameWithVersion + "_METADATA", "w+") as f:
    f.write(formatted_sample_stats)


'''
Instruction:
  Tune LabelGeneratingFunction, then run print_naive_sample_stats to see
  how well the lgf generates training data labels.
  When happy with lgf, run write_naive_samples_to_data_file to create
  NWD (naive weak dataset).
'''
input_path = '../imdb-data/og/'
output_path = '../imdb-data/'
#print_naive_sample_stats(input_path)
write_naive_samples_to_data_file(input_path, output_path)
