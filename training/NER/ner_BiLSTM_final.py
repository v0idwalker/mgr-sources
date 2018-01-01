from collections import Counter
import gensim
import numpy

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense
from keras.utils.vis_utils import plot_model

w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

N_OF_LAB_CLASSES = 10 # Number of classes in trainig data
N_OF_FEAT = 300 # for W2V, 100 for GLOVE. dimensionality of vector space.

def gimme_rand_wv():
    new_random_wv = numpy.array((numpy.random.rand(N_OF_FEAT) * 2) - 1, dtype=numpy.float32)
    norm_const = numpy.linalg.norm(new_random_wv)
    new_random_wv /= norm_const
    return new_random_wv

f = open("training/NER/data.txt", "r+", encoding='UTF-8')

count = Counter()

wvm = dict() # word ID -> vector mapping

raw_data = [] # sentences + tags
raw_sentence = [] # sentence only
raw_labels = [] # label only

# word dictionaries
str_id = dict() # word to ID
id_str = dict() # ID to word

# label dictionaries
id_lab = dict() # ID to label_representation
lab_id = dict() # label_representation to ID
onehotvec = dict()

longest_sentence = 0 # padding length


for l in f:
    if l.strip() == "":
        raw_data.append((tuple(raw_sentence), tuple(raw_labels)))
        if longest_sentence < len(raw_sentence):
            longest_sentence = len(raw_sentence)
        raw_sentence = []
        raw_labels = []
    else:
        line = l.split("\t")
        raw_sentence.append(line[0].strip())
        raw_labels.append(line[1].strip())
        if line[0].strip() not in str_id:
            str_id[line[0].strip()] = len(str_id)
            id_str[len(id_str)] = line[0].strip()
        if line[1].strip() not in lab_id:
            it = len(lab_id)
            lab_id[line[1].strip()] = it
            id_lab[it] = line[1].strip()
            one_hot_vec = numpy.zeros(N_OF_LAB_CLASSES, dtype=numpy.int32) # create representation in form of one hot vector
            one_hot_vec[it] = 1
            onehotvec[it] = tuple(one_hot_vec) # store representation
            onehotvec[tuple(one_hot_vec)] = it

# adding another representation for PADDING label
it = len(lab_id)
lab_id['NULL'] = it
id_lab[it] = 'NULL'
one_hot_vec = numpy.zeros(N_OF_LAB_CLASSES, dtype=numpy.int32)
one_hot_vec[N_OF_LAB_CLASSES-1] = 1
onehotvec[it] = tuple(one_hot_vec)
onehotvec[tuple(one_hot_vec)] = it

# generating wordvectors from word2vec pretrained
# known from w2v
# unknown are randomised w2v like 300 dimensional word-vectors
for word in str_id:
    try:
        wvm[str_id[word]] = w2v.wv[word]
    except KeyError:
        wvm[str_id[word]] = gimme_rand_wv()
        # print(word)

# building data from
# print(raw_data[:20])

X = [] # all data
Y = [] # all labels


for sentence, labels in raw_data:
    wv, lab = [], []

    for itx in range(len(sentence)):
        single_wv = sentence[itx]
        single_lab = labels[itx]
        wv.append(wvm[str_id[single_wv]]) # appending the wordvectors
        lab.append(list(onehotvec[lab_id[single_lab]])) # appending the label representation

    # padding the sequences
    pad_X = numpy.zeros(300)
    pad_Y = numpy.array(onehotvec[lab_id['NULL']])
    pad_length = longest_sentence - len(wv)
    X.append(((pad_length) * [pad_X]) + wv)
    Y.append(((pad_length) * [pad_Y]) + lab)

dX = numpy.array(X)
dY = numpy.array(Y)

perc = 90 # diff between taina nd test

test_split_mask = numpy.random.rand(len(dX)) < (0.01*perc)
# print(test_split_mask[:10])
# print(~test_split_mask[:10])
train_X = dX[test_split_mask]
train_Y = dY[test_split_mask]
test_X = dX[~test_split_mask]
test_Y = dY[~test_split_mask]

# defining deep NER model
model = Sequential()

model.add(Bidirectional(LSTM(units=150, return_sequences=True), input_shape=(29,300)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(10, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_X, train_Y, epochs=25, batch_size=56, validation_data=(test_X, test_Y))

score = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy on test: " + str(score[1]))

import pydot
import graphviz

plot_model(model, to_file='BiLSTMv1.png', show_layer_names=True)