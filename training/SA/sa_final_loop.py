import nltk
import re
import codecs
from collections import Counter
import random
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

count = Counter()
vocab = {}
data = []
labels = []
i = 0

for l in text_pos:
    # print(l)
    l = sanitise_text(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    sentence = []
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            sentence.append(w)
    data.append(sentence)
    labels.append(1)

for l in text_neg:
    # print(l)
    l = sanitise_text(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    sentence = []
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and useless strings
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            sentence.append(w)
    data.append(sentence)
    labels.append(0)

# print(longest)
# print(vocab)
# print(len(count))
# the longest sentence in the data is 62 character long ~64 is nicer

# print(count.most_common())
# creating vocabulary from all the words in data
it = 0
vocab = dict()
for tupl in count.most_common():
    vocab[tupl[0]] = it
    vocab[it] = tupl[0]
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
    "embed_dims": 128,
    "filters": 16,
    "filter_size": 4,
    "hidden_dims": 64,
    "epochs": 25
}

print('Data is being distributed into train/test sets')

def print_predict(prediction, test_dat, test_lab, vocab):
    for j, p in enumerate(prediction):
        print(str(j)+': '+str(round(p[0], 3)), end=' ')
        print(test_lab[j])
        for w in test_dat[j]:
            if w != 0:
                print(vocab.get(w), end=' ')
        print('\n')

# print(model.summary())
# print_predict(pred, test_data, test_labels, vocab)

def plot_graph_from_hist(histories):
    import matplotlib.pyplot as plt
    import datetime, time
    # summarize keras history for accuracy
    for h in histories:
        plt.plot(h.history['acc'], color='red', linestyle='solid', label="Train accuracy")
        plt.plot(h.history['val_acc'], color='blue', linestyle='solid', label="Validation accuracy")
        plt.plot(h.history['loss'], color='red', linestyle='dashed', label="Test loss")
        plt.plot(h.history['val_loss'], color='blue', linestyle='dashed', label="Test loss")
    plt.title('model accuracy/loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='upper left')
    # plt.show()
    plt.savefig('plots/SA_fig'+str(time.mktime(datetime.datetime.today().timetuple()))+'.png')

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=param["max_feat"]) #(train_data, train_)
histories = []

for x in range(0, 20):

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for (d, lab) in zip(id_data, labels):
        if (random.randint(1,100)<=perc):
            # traindata
            train_data.append(d)
            train_labels.append(lab)
        else:
            # testdata
            test_data.append(d)
            test_labels.append(lab)

    # print(imdb.load_data(num_words=param["max_feat"]))
    # print("\n\n")
    # print(x_train)
    # print(y_train)

    print(len(train_data), 'train sequences')
    print(len(test_data), 'test sequences')

    print('Pad data to be uniformly long (samples length x time)')
    train_data = sequence.pad_sequences(train_data, maxlen=param["max_len"])
    test_data = sequence.pad_sequences(test_data, maxlen=param["max_len"])
    print('x_train shape:', train_data.shape)
    print('x_test shape:', test_data.shape)


    # print('Build model...')

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(param["max_feat"],
                        param["embed_dims"],
                        input_length=param["max_len"]))
    # model.add(Dropout(0.3)) # 0.2

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(param["filters"],
                     param["filter_size"],
                     padding='valid',
                     activation='relu',
                     kernel_regularizer=regularizers.l2(l=0.004),
                     # bias_regularizer=regularizers.l2(l=0.001),
                     # activity_regularizer=regularizers.l2(l=0.001),
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(param["hidden_dims"]))
    model.add(Dropout(0.5)) # 0.1
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.003)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    histories.append(model.fit(train_data, train_labels,
              batch_size=param["batch_size"],
              epochs=param["epochs"],
              validation_data=(test_data, test_labels)))
    # pred = model.predict_proba(test_data, verbose=1)
    score, acc = model.evaluate(test_data, test_labels,
                                verbose=1)
    print('Test accuracy:', acc, 'Test score: ', score)
    # print(pred)



plot_graph_from_hist(histories)
wh_acc = 0
wh_loss = 0
wh_vacc = 0
wh_vloss = 0

for h in histories:
    wh_acc += h.history['acc'][len(h.history['acc'])-1]
    wh_vacc += h.history['val_acc'][len(h.history['val_acc'])-1]
    wh_loss += h.history['loss'][len(h.history['loss'])-1]
    wh_vloss += h.history['val_loss'][len(h.history['val_loss'])-1]

print('acc:'+str(wh_acc/20))
print('val_acc:'+ str(wh_vacc/20))
print('loss:' +str(wh_loss/20))
print('val_loss:' + str(wh_vloss/20))

# plot model

# plot_model(model, to_file='model.png')
#
# model.save("textcnn_sa.h5")
# model.save_weights("textcnn_weights_sa.h5")
