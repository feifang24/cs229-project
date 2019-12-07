from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

from dataprocessor import ImdbProcessor
import lstm_preprocess

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
dataDir = '../imdb-data'
dataProcessor = ImdbProcessor(dataDir)
trainExamples, devExamples = dataProcessor.get_train_and_dev_examples('og')
testExamples = dataProcessor.get_test_examples()
allExamples = trainExamples + devExamples
dictionary, reverseDictionary = lstm_preprocess.construct_dictionaries(allExamples)

xTrain = lstm_preprocess.examples_to_matrix(allExamples, dictionary, maxlen)
yTrain = lstm_preprocess.labels_to_vector([example.label for example in allExamples])
print(xTrain.shape, 'train sequences')
print(yTrain.shape, 'train labels')
xTest = lstm_preprocess.examples_to_matrix(testExamples, dictionary, maxlen)
yTest = lstm_preprocess.labels_to_vector([example.label for example in testExamples])
print(xTest.shape, 'test sequences')
print(yTest.shape, 'test labels')

print('Pad sequences (samples x time)')
xTrain = sequence.pad_sequences(xTrain, maxlen=maxlen)
xTest = sequence.pad_sequences(xTest, maxlen=maxlen)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(xTrain, yTrain,
          batch_size=batch_size,
          epochs=10,
          validation_data=(xTest, yTest))
score, acc = model.evaluate(xTest, yTest,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)