import collections

import numpy as np

from sklearn.model_selection import train_test_split
import re
import os

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = re.sub(r'[^\w\s]','',message.lower())
    return message.split()

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    # create a frequency map
    freq_map = {}

    for message in messages:
        words = set(get_words(message))
        for word in words:
            if word not in freq_map:
                freq_map[word] = 0
            freq_map[word] += 1

    # get list of frequent words
    min_occurrence = 5
    frequent_words = [word for word, frequency in freq_map.items()
                         if frequency >= min_occurrence]
    return {word: i for i, word in enumerate(frequent_words)}


    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    def create_message_entry(message):
        words = get_words(message)
        entry = [0] * len(word_dictionary)
        for word in words:
            if word in word_dictionary.keys():
                entry[word_dictionary[word]] += 1
        return entry

    # returns np array of shape (n_messages, dict_size)
    return np.asarray([create_message_entry(message) for message in messages])

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n_examples, dict_size = matrix.shape

    probs_y = sum(labels) / len(labels)

    def get_word_pos(k):
        num = np.dot(matrix[:, k], labels) + 1
        den = np.dot(np.sum(matrix, axis=1), labels) + dict_size
        return num / den

    def get_word_neg(k):
        num = np.dot(matrix[:, k], 1 - labels) + 1
        den = np.dot(np.sum(matrix, axis=1), 1 - labels) + dict_size
        return num / den

    probs_pos = [get_word_pos(k) for k in range(dict_size)]
    probs_neg = [get_word_neg(k) for k in range(dict_size)]
    return {'probs_y': probs_y, 'probs_pos': probs_pos, 'probs_neg': probs_neg}
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    probs_y = model['probs_y']
    probs_pos = model['probs_pos']
    probs_neg = model['probs_neg']

    log_probs_pos = np.log(np.asarray(probs_pos))
    log_probs_neg = np.log(np.asarray(probs_neg))

    # let statistic be log(numerator of pos prob) - log(numerator of neg prob)
    statistic = np.log(probs_y / (1 - probs_y)) + np.matmul(matrix, log_probs_pos - log_probs_neg)
    return np.array(statistic > 0).astype(int)

    # *** END CODE HERE ***


def sort_indicative_keywords(model, dictionary):
    probs_pos = model['probs_pos']
    probs_neg = model['probs_neg']

    log_probs_pos = np.log(np.asarray(probs_pos))
    log_probs_neg = np.log(np.asarray(probs_neg))

    indication = log_probs_pos - log_probs_neg

    #argsort indication, get the last five indices (they are the most indicative)
    most_neg_indices = np.argsort(indication)
    most_pos_indices = most_neg_indices[::-1]

    return most_pos_indices, most_neg_indices


def return_keywords_indices(data):
    # data is a list of 2-elt lists [text, label]
    all_messages = [d[0] for d in data] 
    all_labels = np.asarray([d[1] for d in data])

    train_messages, test_messages, train_labels, test_labels = train_test_split(
        all_messages, all_labels, test_size=0.2, random_state=42)

    dictionary = create_dictionary(train_messages)

    words = list(dictionary.keys())

    print('Size of dictionary: ', len(dictionary))

    train_matrix = transform_text(train_messages, dictionary)

    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    most_pos_indices, most_neg_indices = sort_indicative_keywords(naive_bayes_model, dictionary)

    return words, most_pos_indices, most_neg_indices


def main():
    all_messages, all_labels = util.read_data('../../imdb-data/sd1600')
    train_messages, test_messages, train_labels, test_labels = train_test_split(
        all_messages, all_labels, test_size=0.2, random_state=42)

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('imdb_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('imdb_sample_train_matrix', train_matrix[:100,:])

    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('imdb_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_pos_words, top_neg_words = get_top_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top positive words for Naive Bayes are: ', top_pos_words)

    print('The top negative words for Naive Bayes are: ', top_neg_words)

    util.write_json('imdb_top_indicative_words', top_pos_words + top_neg_words)



if __name__ == "__main__":
    main()
