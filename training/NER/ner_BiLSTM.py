from collections import Counter
import random
import gensim
import numpy

from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dropout, Dense
from keras.utils.vis_utils import plot_model

# w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# print(w2v.wv['man'])

f = open("training/NER/data.txt", "r+", encoding='UTF-8')


longest = 0     # longest sentence in the data

count = Counter()
# word -> vector mapping
wvm = dict()

# word dictionaries
str_id = dict()
id_str = dict()

# label dictionaries
id_lab = dict()
lab_id = dict()

for l in f:
    if l.strip() == "":
        pass
    else:
        line = l.split("\t")
        if line[0].strip() not in str_id:
            str_id[line[0].strip()] = len(str_id)
            id_str[len(id_str)] = line[0].strip()
            # print(l.split("\t")[0].strip())
        if line[1].strip() not in lab_id:
            lab_id[line[1].strip()] = len(lab_id)
            id_lab[len(id_lab)] = line[1].strip()
            # print(l.split("\t")[1].strip())


# print(str_id.keys())
# print(lab_id.keys())

# reiterate over data

f.seek(0, 0)

data = []
labels = []
sentence = []
sent_lab = []

for l in f:
    # print(l) # do not use in windows or any other crippled consoles, as it breaks, thanks to windows console not being unicode, or being unable to show some unicode chars.
    if l.strip() == "":
        # store data in selected arrays
        data.append(sentence)
        labels.append(sent_lab)
        sentence = []
        sent_lab = []
    else:
        line = l.split("\t")
        sentence.append(str_id[line[0].strip()])
        sent_lab.append(lab_id[line[1].strip()])

# print(data[:20])
# print(str_id)
# print(labels[:20])
# print(lab_id)
# print(id_lab)

f.close()
unknown = 0
# generating wordvectors
for word in str_id:
    try:
        # wvm[word] = w2v.wv[word]
        pass
    except KeyError:
        wvm[word] = (numpy.random.rand(300)*2)-1
        unknown = unknown+1
        # print(word)

# print(str(unknown) + ' ' +str(len(str_id)) )

from data_preprocess import DataUtil

dutil = DataUtil("wordvecs.txt", "news_tagged_data.txt")
dX, dY = dutil.read_and_parse_data("wordvecs.txt", "news_tagged_data.txt")

perc = 90

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