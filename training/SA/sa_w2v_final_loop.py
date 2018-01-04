import nltk
import codecs
from collections import Counter
import random
import os
import gensim as gs
import numpy
import helper


# we want to find the Named Entities in the text
# then select all the sentences containing the entities and get their respective sentiment
# show the words with their respective prevalent sentiment used in the text

# the interesting windows is a sentence
# the context of the whole scope of the text is not interesting


nltk.download('stopwords')  # only in case of english texts could this help
nltk.download('punkt')

# create vocabulary from training files/words/
text_pos = codecs.open(os.path.join('training','SA','sentence_polarity','rt-polaritydata', 'rt-polarity.pos'), "r", "ISO-8859-1")
text_neg = codecs.open(os.path.join('training','SA','sentence_polarity','rt-polaritydata', 'rt-polarity.neg'), "r", "ISO-8859-1")

stopwords = set(nltk.corpus.stopwords.words('english'))
longest = 0     # longest sentence in the data

count = Counter()
vocab = {}
data = []
labels = []
i = 0

for l in text_pos:
    # print(l)
    l = helper.sanitise_text(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    sentence = []
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        if (w.lower() not in stopwords) and (helper.sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            sentence.append(w)
    data.append(sentence)
    labels.append(1)

for l in text_neg:
    # print(l)
    l = helper.sanitise_text(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    sentence = []
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        if (w.lower() not in stopwords) and (helper.sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            sentence.append(w)
    data.append(sentence)
    labels.append(0)

# print(longest)
# print(vocab)
# print(len(count))
# the longest sentence in the data is 62 character long ~64 is rounded

# print(count.most_common())
# creating vocabulary from all the words in data
it = 0
vocab = dict()
for tupl in count.most_common():
    vocab[tupl[0]] = it
    it += 1

# for v in vocab.items():
#     print(v)

id_data = []
for d in data:
    sentence = []
    for w in d:
        sentence.append(vocab[w])
    id_data.append(sentence)

# print(id_data)

# for hyperas/hyperopt to work, we need to define the functions as follows, otherwise the data has to be loaded every
# time a run is completed

# 4-5k should be enough as after that, there are words that do not have enough instances in the text

# keras trainig

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.utils import plot_model
from keras import optimizers, regularizers
from sklearn.model_selection import KFold

# create param space, test with cutting out the most used words as well as least used, as the former tend to appear too
# often, while the latter has too few instances.

perc = 90  # the ratio of training data compared to test data 80~72 90~75

param = {
    "max_len": 64,
}

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=param["max_feat"]) #(train_data, train_)

# creating w2v embedding layer
w2v = gs.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# assemble the embedding_weights in one numpy array
vocab_dim = 300 # dimensionality of your word vectors, just as 300 dims in w2v pre-trained
n_symbols = len(vocab) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = numpy.zeros((n_symbols, vocab_dim))
for word, index in vocab.items():
    try:
        embedding_weights[index, :] = w2v.wv[word]
    except KeyError:
        # embedding_weights[index, :] = numpy.array(numpy.zeros(300), dtype=float) # add random instead of zeroes, might get better success rates. Faster learning rate if not random.
        new_random_wv = numpy.array((numpy.random.rand(300) * 2) - 1, dtype=numpy.float32)
        norm_const = numpy.linalg.norm(new_random_wv)
        new_random_wv /= norm_const
        embedding_weights[index, :] = new_random_wv

# define inputs here

histories = []
accu = []
runs = 10

for x in range(0, runs):

    print('Data is being distributed into train/test sets')
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    embedding_layer = Embedding(output_dim=vocab_dim, input_dim=n_symbols, trainable=True)
    embedding_layer.build((None,))  # if you don't do this, the next step won't work
    embedding_layer.set_weights([embedding_weights])

    for (d, l) in zip(id_data, labels):
        if (random.randint(1, 100) <= perc):
            # traindata
            train_data.append(d)
            train_labels.append(l)
        else:
            # testdata
            test_data.append(d)
            test_labels.append(l)

    print(len(train_data), 'train sequences')
    print(len(test_data), 'test sequences')

    print('Pad data to be uniformly long (samples length x time)')
    train_data = sequence.pad_sequences(train_data, maxlen=param["max_len"])
    test_data = sequence.pad_sequences(test_data, maxlen=param["max_len"])
    print('x_train shape:', train_data.shape)
    print('x_test shape:', test_data.shape)

    param = {
        "max_len": 64,
        "batch_size": 32, #16?
        "embed_dims": 128,
        "filters": 16,
        "filter_size": 4,
        "hidden_dims": 64,
        "epochs": 10
    }

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions

    model.add(embedding_layer)

    # {'Conv1D': 128, 'batch_size': 96, 'Conv1D_3': 4, 'Dropout_1': 0.3, 'Dropout': 0.4, 'Conv1D_1': 4, 'Conv1D_2': 64, 'Dense': 32}
    # Trial 0 vals: {'Conv1D_1': [1], 'batch_size': [2], 'Dense': [0], 'Conv1D': [1], 'Dropout': [0.28192006496913374]}
    # {'Conv1D_1': 8, 'batch_size': 96, 'Dense': 32, 'Conv1D': 16, 'Dropout': 0.28192006496913374}
    # {'Conv1D': 128, 'Dropout': 0.37678000665362027, 'Conv1D_1': 12, 'batch_size': 32, 'Dense': 32}

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(32,
                     6,
                     padding='valid',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(l=0.01),
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(64))
    model.add(Dropout(0.5)) # 0.1
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0058)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    histories.append(model.fit(train_data, train_labels,
              batch_size=32,
              epochs=param["epochs"],
              validation_data=(test_data, test_labels)))

    score, acc = model.evaluate(test_data, test_labels, verbose=1)
    accu.append(acc)
    print('Test accuracy:', acc, 'Test score: ', score)

helper.plot_graph_from_hist(histories, filepath='plots', filename='SA_loop_final')

mean_acc = 0

for a in accu:
    mean_acc += a

print('val_acc:'+str(mean_acc/runs))

# plot model

# plot_model(model, to_file='model.png')

# model.save("textcnn_sa_w2v.h5")
# model.save_weights("textcnn_weights_sa_w2v.h5")
