import nltk
import re
import codecs
from collections import Counter
import sys
import os

# we want to find the Named Entities in the text
# then select all the sentences containing the entities and get their respective sentiment
# show the words with their respective prevalent sentiment used in the text

# the interesting windows is a sentence
# the context of the whole scope is or is not interesting


nltk.download('stopwords')  # only in case of english texts could this help


def sanitise_text_for_tf(string):
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # english pre-processing for TF
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def sanitise_vocabulary_string(string):
    if string not in [".", ",", "(", ")", "[", "]", "\"", "--", "'", "``", "`", "'s", "n't"]:
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
    # https://en.wiktionary.org/wiki/Category:English_words_suffixed_with_-n%27t
    # isn't useful or is?
    return string

# create vocabulary from training files/words/

text_pos = codecs.open("training/SA/sentence_polarity/rt-polaritydata/rt-polarity.pos", "r", "ISO-8859-1")
text_neg = codecs.open("training/SA/sentence_polarity/rt-polaritydata/rt-polarity.neg", "r", "ISO-8859-1")

stopwords = set(nltk.corpus.stopwords.words('english'))

longest = 0

count = Counter()
vocab = {}
i = 0

for l in text_pos:
    # print(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords!!
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            if w.lower() not in vocab:
                vocab[i] = w.lower()
                i += 1

for l in text_neg:
    # print(l)
    if (longest < len(nltk.word_tokenize(l, 'english'))):
        longest = len(nltk.word_tokenize(l, 'english'))
    for w in nltk.word_tokenize(l, 'english'):
        # remove stopwords and uslesee strings
        if (w.lower() not in stopwords) and (sanitise_vocabulary_string(w.lower())):
            count[w.lower()] += 1
            if w.lower() not in vocab:
                vocab[i] = w.lower()
                i += 1

#print(longest)
# print(vocab)

# for item in count.items(): print("{}\t{}".format(*item))
print(i)
print(count.most_common(4000))