import nltk
import re
import codecs
from collections import Counter
import sys
import os

# we want to find the Named Entities in the text
# then select all the sentences containing the entities and get their respective sentiment
# show the words with their respective prevalent sentiment used in the text

# the interesting windows is a sentence
# the context of the whole scope is or is not interesting


nltk.download('stopwords')  # only in case of english texts could this help


def sanitise_tokenize_text(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # english pre-processing for TF
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"-", " - ", string)
    return string.strip().lower()


def sanitise_vocabulary_string(string):
    # weird
    if string not in [".", ",", "(", ")", "[", "]", "\"", "--", "'", "``", "`"]:
        # , "'s", "n't" ??
        return True
    else:
        return False


def sanitise_en_negation(string):
    # replace negation with reasonable words
    string.replace("can\'t", "can not")
    string.replace("couldn\'t", "could not")
    string.replace("won\'t", "will not")
    string.replace("wouldn\'t", "would not")
    string.replace("isn\'t", "is not")
    string.replace("wasn\'t", "was not")
    string.replace("shan\'t", "shall not")
    string.replace("shouldn\'t", "should not")
    # ...
    # https://en.wiktionary.org/wiki/Category:English_words_suffixed_with_-n%27t
    # isn't useful or is?
    return string


# create vocabulary from training files/words/

text_pos = codecs.open("training/SA/sentence_polarity/rt-polaritydata/rt-polarity.pos", "r", "ISO-8859-1")
text_neg = codecs.open("training/SA/sentence_polarity/rt-polaritydata/rt-polarity.neg", "r", "ISO-8859-1")

stopwords = set(nltk.corpus.stopwords.words('english'))

longest = 0

count = Counter()
vocab = {}
i = 0

for l in text_pos:
    # print(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        w = sanitise_tokenize_text(w)
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            # if w.lower() not in vocab:
            #     vocab[i] = w.lower()
            #     i += 1

for l in text_neg:
    # print(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            # if w.lower() not in vocab:
            #     vocab[i] = w.lower()
            #     i += 1

# print(longest)
# print(vocab)
# print(len(count))
# the longest sentence in the data is 62 character long ~64 is nicer

# print(count.most_common())
# creating vocab
it = 0
vocab = dict()
for tupl in count.most_common():
    vocab[it] = tupl
    it += 1

for v in vocab.items():
    print(v)

# todo: break - interconnected words apart

# 4-5k should be enough as after that, there are words that do not have enough instances in the text

# keras trainig

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import plot_model

from keras.datasets import imdb

# create param space, test with cutting out the most used words as well as least used, as the former tend to appear too
# often, while the latter has too few instances.

# perc = 0.7 # the ratio of training data compared to test data
# param = {
#     "max_feat": 5000,
#     "max_len": 64,
#     "batch_size": 32,
#     "embed_dims": 50,
#     "filters": 250,
#     "kernel_size": 3,
#     "hidden_dims": 250,
#     "epochs": 4
# }
#
# print('Data is being distributed into train/test sets')
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=param["max_feat"]) #(train_data, train_)
#
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad data to be "rectangular" (samples lenght x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=param["max_len"])
# x_test = sequence.pad_sequences(x_test, maxlen=param["max_len"])
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
#
# print('Build model...')
#
# model = Sequential()
#
# # we start off with an efficient embedding layer which maps
# # our vocab indices into embedding_dims dimensions
# model.add(Embedding(param["max_feat"],
#                     param["embed_dims"],
#                     input_length=param["max_len"]))
# model.add(Dropout(0.2))
#
# # we add a Convolution1D, which will learn filters
# # word group filters of size filter_length:
# model.add(Conv1D(param["filters"],
#                  param["kernel_size"],
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# # we use max pooling:
# model.add(GlobalMaxPooling1D())
#
# # We add a vanilla hidden layer:
# model.add(Dense(param["hidden_dims"]))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
#
# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=param["batch_size"],
#           epochs=param["epochs"],
#           validation_data=(x_test, y_test))
#
# # plot model
#
# plot_model(model, to_file='model.png')
