import nltk
import re
import codecs
from collections import Counter
import random

import sys
import os

# we want to find the Named Entities in the text
# then select all the sentences containing the entities and get their respective sentiment
# show the words with their respective prevalent sentiment used in the text

# the interesting windows is a sentence
# the context of the whole scope of the text is not interesting


nltk.download('stopwords')  # only in case of english texts could this help
nltk.download('punkt')


def sanitise_text(string):
    # english regexp pre-processing for TF
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string) # weird string removal
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"-", " - ", string)
    string = re.sub(r"'", " \" ", string)
    string = re.sub(r"\"", " \" ", string)
    return string.strip().lower()


def sanitise_vocabulary_string(string):
    # weird
    if string not in ["--", "``", "`"]:
        # , "'s", "n't", ".", ",", "(", ")", "[", "]", "'", "\"", ??
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
text_pos = codecs.open(os.path.join('training','SA','sentence_polarity','rt-polaritydata', 'rt-polarity.pos'), "r", "ISO-8859-1")
text_neg = codecs.open(os.path.join('training','SA','sentence_polarity','rt-polaritydata', 'rt-polarity.neg'), "r", "ISO-8859-1")

stopwords = set(nltk.corpus.stopwords.words('english'))
longest = 0     # longest sentence in the data

from keras.datasets import imdb
from keras.preprocessing import sequence

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=6000,
                                                      skip_top=0,
                                                      maxlen=1600,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

# keras trainig

maxlen = 100
maxfeat =20000

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import plot_model

# create param space, test with cutting out the most used words as well as least used, as the former tend to appear too
# often, while the latter has too few instances.

perc = 90  # the ratio of training data compared to test data 80~72 90~75
# baseparam = {
#     "max_feat": 6000,
#     "max_len": 64,
#     "batch_size": 128,
#     "embed_dims": 50,
#     "filters": 24,
#     "kernel_size": 4,
#     "hidden_dims": 128,
#     "epochs": 10
# }
#
# for

param = {
    "max_feat": 6000,
    "max_len": 64,
    "batch_size": 32, #16?
    "embed_dims": 300,
    "filters": 16,
    "filter_size": 4,
    "hidden_dims": 64,
    "epochs": 2 #20
}

# print('Data is being distributed into train/test sets')


# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=param["max_feat"]) #(train_data, train_)

average = 0

# print('Build model...')
for i in range(0, 10):

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(maxfeat,
                        param["embed_dims"],
                        ))
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(64,
                     4,
                     padding='valid',
                     activation='relu',
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dropout(0.3)) # 0.2
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train,
              batch_size=param["batch_size"],
              epochs=param["epochs"],
              validation_data=(x_test, y_test))
    pred = model.predict_proba(x_test, verbose=1)
    score, acc = model.evaluate(x_test, y_test,
                                verbose=1)
    print('Test accuracy:', acc, 'Test score: ', score)
    # print(pred)

    average += acc

    def print_predict(prediction, test_dat, test_lab, vocab):
        for j, p in enumerate(prediction):
            print(str(j)+': '+str(round(p[0], 3)), end=' ')
            print(test_lab[j])
            for w in test_dat[j]:
                if w != 0:
                    print(vocab.get(w), end=' ')
            print('\n')

    print(model.summary())
# print_predict(pred, test_data, test_labels, vocab)

# plot model

# plot_model(model, to_file='model.png')

# model.save("textcnn_sa_van.h5")
# model.save_weights("textcnn_weights_sa_van.h5")

print(average/10)
