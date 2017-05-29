# data preparation for training
from collections import Counter
import random

f = open("training/NER/data.txt", "r+")

longest = 0     # longest sentence in the data

count = Counter()

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
    # print(l)
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

print(data[:20])
# print(str_id)
print(labels[:20])
# print(lab_id)
print(id_lab)

f.close()

perc = 80
max_len = 0
train_data = []
test_data = []
train_labels = []
test_labels = []

for d in data:
    if max_len < len(d):
        max_len = len(d)

for (d, l) in zip(data, labels):
    if (random.randint(1, 100) <= perc):
        # traindata
        train_data.append(d)
        train_labels.append(l)
    else:
        # testdata
        test_data.append(d)
        test_labels.append(l)

# train data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional

max_feat = 30000
embedding_dimensions = 128
epochs = 10
batch_size = 64

train_data = sequence.pad_sequences(data, maxlen=max_len)
train_labels = sequence.pad_sequences(labels, maxlen=max_len)

print('training data shape:', train_data.shape)
print('testing data shape:', test_data.shape)

model = Sequential()

# model.add(Embedding())
# add the first layer with input_shape
model.add(Embedding(max_feat,
                    embedding_dimensions,
                    input_length=max_len))


# model.add(Conv1D(3,
#                  8,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
#
# model.add(GlobalMaxPooling1D())
#
# model.add(Dense(256))
# model.add(Dropout(0.5)) # 0.1
# model.add(Activation('relu'))
#
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

model.add(Bidirectional(LSTM(64, return_sequences=True)))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_data, test_labels))
