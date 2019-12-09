import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import lstm_preprocess
from dataprocessor import ImdbProcessor
from tokenizer import FullTokenizer

# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
remove_top_words = 0
max_review_length = 500
K = top_words - 1
embedding_vecor_length = 32

dataDir = '../imdb-data'
dataProcessor = ImdbProcessor(dataDir)
tokenizer = FullTokenizer("vocab.txt")

devExamples = dataProcessor.get_dev_examples()
testExamples = dataProcessor.get_test_examples()
posTestExamples, negTestExamples = dataProcessor.get_test_examples(True)

for trainingSet in ['og', 'sd800', 'sd1600', 'sd3200', 'sd6400', 'sd12800', 'nwd00', 'nwd01']:
  trainExamples = dataProcessor.get_train_examples(trainingSet)
  allExamples = trainExamples + devExamples
  dictionary, reverseDictionary = lstm_preprocess.construct_dictionary(allExamples, tokenizer, K, remove_top_words)

  xTrain = lstm_preprocess.examples_to_list_of_indices(trainExamples, tokenizer, dictionary)
  xDev = lstm_preprocess.examples_to_list_of_indices(devExamples, tokenizer, dictionary)
  xTest = lstm_preprocess.examples_to_list_of_indices(testExamples, tokenizer, dictionary)
  xPosTest = lstm_preprocess.examples_to_list_of_indices(posTestExamples, tokenizer, dictionary)
  xNegTest = lstm_preprocess.examples_to_list_of_indices(negTestExamples, tokenizer, dictionary)
  yTrain = lstm_preprocess.labels_to_vector([example.label for example in trainExamples])
  yDev = lstm_preprocess.labels_to_vector([example.label for example in devExamples])
  yTest = lstm_preprocess.labels_to_vector([example.label for example in testExamples])
  yPosTest = lstm_preprocess.labels_to_vector([example.label for example in posTestExamples])
  yNegTest = lstm_preprocess.labels_to_vector([example.label for example in negTestExamples])
  # truncate and pad input sequences

  xTrain = sequence.pad_sequences(xTrain, maxlen=max_review_length)
  xDev = sequence.pad_sequences(xDev, maxlen=max_review_length)
  xTest = sequence.pad_sequences(xTest, maxlen=max_review_length)
  xPosTest = sequence.pad_sequences(xPosTest, maxlen=max_review_length)
  xNegTest = sequence.pad_sequences(xNegTest, maxlen=max_review_length)

  # create the model
  model = Sequential()
  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(LSTM(100))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  model.fit(xTrain, yTrain, epochs=6, batch_size=64, validation_data=(xDev, yDev))
  # Final evaluation of the model
  testScores = model.evaluate(xTest, yTest, verbose=0)
  posTestScores = model.evaluate(xPosTest, yPosTest, verbose=0)
  negTestScores = model.evaluate(xNegTest, yNegTest, verbose=0)
  out = "Test Accuracy: %.2f%%\n" % (testScores[1]*100)
  out += "Pos Accuracy: %.2f%%\n" % (posTestScores[1]*100)
  out += "Neg Accuracy: %.2f%%\n" % (negTestScores[1]*100)
  with open(trainingSet + "_stat.txt", "w+") as f:
    f.write(out)