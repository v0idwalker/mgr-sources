# data preparation for training
from collections import Counter
import random

f = open("data.txt", "r+")

longest = 0     # longest sentence in the data

count = Counter()
strid = dict()
idstr = dict()
data = []
labels = []
lab_str = [] # make 2 dict again
i = 0
j = 0

for l in f:
    if l.strip() == "":
        i += 1
        if longest < j:
            longest = j
    else:
        j += 1
        if l.split(" ")[0] not in strid:
            strid[l.split(" ")[0].strip()] = len(strid)
            idstr[len(idstr)] = l.split(" ")[0].strip()
        if l.split(" ")[1] not in lab_str:
            lab_str.add(l.split(" ")[1].strip())


from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D

model = Sequential()

# model.add(Embedding())

model.add(Conv1D(3,
                 8,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())

model.add(Conv1D(6,
                 32,
                 padding='valid',
                 activation='relu',
                 strides=1))

model.add(GlobalMaxPooling1D())

model.add(Dense(256))
model.add(Dropout(0.5)) # 0.1
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(train_data, train_labels, batch_size=param["batch_size"], epochs=param["epochs"],
#           validation_data=(test_data, test_labels))