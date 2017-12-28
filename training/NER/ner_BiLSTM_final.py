from collections import Counter
import gensim
import numpy

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense
from keras.utils.vis_utils import plot_model

w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# print(w2v.wv['man'])

f = open("training/NER/data.txt", "r+", encoding='UTF-8')


longest = 0     # longest sentence in the data

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


for l in f:
    if l.strip() == "":
        raw_data.append(tuple(raw_sentence), tuple(raw_labels))
        raw_sentence = []
        raw_labels = []
    else:
        line = l.split("\t")
        raw_sentence.append(line[0].strip())
        raw_labels.append(line[1].strip())
        if line[0].strip() not in str_id:
            str_id[line[0].strip()] = len(str_id)
            id_str[len(id_str)] = line[0].strip()
            # print(l.split("\t")[0].strip())
        if line[1].strip() not in lab_id:
            lab_id[line[1].strip()] = len(lab_id)
            id_lab[len(id_lab)] = line[1].strip()
            # print(l.split("\t")[1].strip())



unknown = 0
# generating wordvectors
for word in str_id:
    try:
        wvm[str_id[word]] = w2v.wv[word]
        pass
    except KeyError:
        new_random_wv = numpy.array((numpy.random.rand(300) * 2) - 1, dtype=numpy.float32)
        norm_const = numpy.linalg.norm(new_random_wv)
        new_random_wv /= norm_const
        wvm[str_id[word]] = new_random_wv
        # print(word)

# print(str(unknown) + ' ' +str(len(str_id)) )



from data_preprocess import DataUtil

dutil = DataUtil("wordvecs.txt", "news_tagged_data.txt")
dX, dY = dutil.read_and_parse_data("wordvecs.txt", "news_tagged_data.txt")

perc = 90 # diff between taina nd test

test_split_mask = numpy.random.rand(len(dX)) < (0.01*perc)
train_X = dX[test_split_mask]
train_Y = dY[test_split_mask]
test_X = dX[~test_split_mask]
test_Y = dY[~test_split_mask]


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