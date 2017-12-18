from collections import Counter
import random
import gensim
import numpy

w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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

print(data[:20])
# print(str_id)
print(labels[:20])
# print(lab_id)
# print(id_lab)

f.close()
unknown = 0
# generating wordvectors
for word in str_id:
    try:
        wvm[word] = w2v.wv[word]
    except KeyError:
        wvm[word] = numpy.random.rand(300)
        unknown = unknown+1
        print(wvm[word])

print(str(unknown) +' ' +str(len(str_id)) )


# perc = 80
# max_len = 0
# train_data = []
# test_data = []
# train_labels = []
# test_labels = []
#
# for d in data:
#     if max_len < len(d):
#         max_len = len(d)
#
# for (d, l) in zip(data, labels):
#     if (random.randint(1, 100) <= perc):
#         # traindata
#         train_data.append(d)
#         train_labels.append(l)
#     else:
#         # testdata
#         test_data.append(d)
#         test_labels.append(l)


