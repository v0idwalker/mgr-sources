import nltk
import re
import codecs
from collections import Counter
from numpy import random

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import gensim as gs
import numpy

import os

# we want to find the Named Entities in the text
# then select all the sentences containing the entities and get their respective sentiment
# show the words with their respective prevalent sentiment used in the text

# the interesting windows is a sentence
# the context of the whole scope of the text is not interesting


nltk.download('stopwords')  # only in case of english texts could this help
nltk.download('punkt')

# for hyperas/hyperopt to work, we need to define the functions as follows, otherwise the data has to be loaded every
# time a run is completed


# training takes between 1s - 4s per epoch on nv-1060-6G@2088MHz depending on hyperparameters.
# best run:
# {'Dense': 64, 'batch_size': 32, 'Dropout_1': 0.2, 'Dropout': 0.3,
# 'Conv1D': 16, 'Embedding_1': 128, 'Conv1D_1': 4, 'Embedding': 6000}
# 2.25s per epoch.
# depending on the batch size (32 - 2.25s, 16 - 4.5s)
def data():
    from keras.preprocessing import sequence
    from keras.layers import Embedding

    def sanitise_text(string):

        string = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", string)  # weird string removal
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

        if string not in ["--", "``", "`"]:
            return True
        else:
            return False

    def sanitise_en_negation(string):

        string.replace("can\'t", "can not")
        string.replace("couldn\'t", "could not")
        string.replace("won\'t", "will not")
        string.replace("wouldn\'t", "would not")
        string.replace("isn\'t", "is not")
        string.replace("wasn\'t", "was not")
        string.replace("shan\'t", "shall not")
        string.replace("shouldn\'t", "should not")

        return string


    text_pos = codecs.open(os.path.join('training', 'SA', 'sentence_polarity', 'rt-polaritydata', 'rt-polarity.pos'),
                           "r", "ISO-8859-1")
    text_neg = codecs.open(os.path.join('training', 'SA', 'sentence_polarity', 'rt-polaritydata', 'rt-polarity.neg'),
                           "r", "ISO-8859-1")

    stopwords = set(nltk.corpus.stopwords.words('english'))
    longest = 0

    count = Counter()
    data = []
    labels = []

    for l in text_pos:

        l = sanitise_text(l)
        if (longest < len(nltk.word_tokenize(l, 'english'))):
            longest = len(nltk.word_tokenize(l, 'english'))
        sentence = []
        for w in nltk.word_tokenize(l, 'english'):

            if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
                count[w.lower()] += 1
                sentence.append(w)
        data.append(sentence)
        labels.append(1)

    for l in text_neg:

        l = sanitise_text(l)
        if (longest < len(nltk.word_tokenize(l, 'english'))):
            longest = len(nltk.word_tokenize(l, 'english'))
        sentence = []
        for w in nltk.word_tokenize(l, 'english'):

            if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
                count[w.lower()] += 1
                sentence.append(w)
        data.append(sentence)
        labels.append(0)


    it = 0
    vocab = dict()
    for tupl in count.most_common():
        vocab[tupl[0]] = it
        it += 1

    id_data = []
    for d in data:
        sentence = []
        for w in d:
            sentence.append(vocab[w])
        id_data.append(sentence)

    perc = 90

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for (d, l) in zip(id_data, labels):
        if random.randint(1, 100) <= perc:
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
    train_data = sequence.pad_sequences(train_data, maxlen=64)
    test_data = sequence.pad_sequences(test_data, maxlen=64)
    print('x_train shape:', train_data.shape)
    print('x_test shape:', test_data.shape)
    X_train = train_data
    Y_train = train_labels
    X_test = test_data
    Y_test = test_labels


    # creating w2v embedding layer
    w2v = gs.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    # assemble the embedding_weights in one numpy array
    vocab_dim = 300  # dimensionality of your word vectors, just as 300 dims in w2v pre-trained
    n_symbols = len(vocab) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = numpy.zeros((n_symbols, vocab_dim))
    for word, index in vocab.items():
        try:
            embedding_weights[index, :] = w2v.wv[word]
        except KeyError:
            embedding_weights[index, :] = numpy.array((numpy.random.rand(300) * 2) - 1, dtype=float)

    embedding_layer = Embedding(output_dim=vocab_dim, input_dim=n_symbols, trainable=True)
    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_weights])

    return X_train, Y_train, X_test, Y_test, embedding_layer

def model_wrap(X_train, Y_train, X_test, Y_test, embedding_layer):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
    from keras.layers.normalization import BatchNormalization
    from keras import optimizers, regularizers

    param = {
        "max_len": 64,
        "epochs": 15
    }

    model = Sequential()

    model.add(embedding_layer)

    model.add(Conv1D({{choice([16, 24, 32, 40])}},   # 16
                     {{choice([4, 8, 12])}},        # 4
                     padding='valid',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(l=0.001),
                     strides=1))

    model.add(GlobalMaxPooling1D())

    model.add(Dense({{choice([24, 32, 64])}})) # 64
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr={{choice([0.001, 0.0001, 0.0005])}}, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                           decay={{choice([0.001, 0.002, 0.003])}})
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=32,     # 32 16?
              epochs=param["epochs"],
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test,
                                verbose=1)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

from hyperas.utils import eval_hyperopt_space

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model, space = optim.minimize(model=model_wrap,
                                                data=data,
                                                algo=tpe.suggest,
                                                max_evals=80,
                                                trials=trials,
                                                eval_space=True,
                                                return_space=True
                                                )
    X_train, Y_train, X_test, Y_test, embedding_layer = data()

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print('\n\n')

    for t, trial in enumerate(trials):
        vals = trial.get('misc').get('vals')
        print("Trial %s vals: %s" % (t, vals))
        print(eval_hyperopt_space(space, vals))
