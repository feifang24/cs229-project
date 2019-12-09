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
remove_top_words = 20
max_review_length = 500
K = top_words - 1

dataDir = '../imdb-data'
dataProcessor = ImdbProcessor(dataDir)
tokenizer = FullTokenizer("vocab.txt")

trainExamples = dataProcessor.get_train_examples('og')
devExamples = dataProcessor.get_dev_examples()
allExamples = trainExamples + devExamples
dictionary, reverseDictionary = lstm_preprocess.construct_dictionary(allExamples, tokenizer, K, remove_top_words)

xTrain = lstm_preprocess.examples_to_list_of_indices(trainExamples, tokenizer, dictionary)
xDev = lstm_preprocess.examples_to_list_of_indices(devExamples, tokenizer, dictionary)
yTrain = lstm_preprocess.labels_to_vector([example.label for example in trainExamples])
yDev = lstm_preprocess.labels_to_vector([example.label for example in devExamples])
# truncate and pad input sequences

xTrain = sequence.pad_sequences(xTrain, maxlen=max_review_length)
xDev = sequence.pad_sequences(xDev, maxlen=max_review_length)

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(xTrain, yTrain, epochs=10, batch_size=64, validation_data=(xDev, yDev))
# Final evaluation of the model
scores = model.evaluate(xDev, yDev, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))