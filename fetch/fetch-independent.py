# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import sys
from nltk import word_tokenize, sent_tokenize
import csv

# windows specific requirenment for printing out
try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError as e:
    print(e)

website = sys.argv[1]

r = requests.get(website)
bsc = BeautifulSoup(r.content, "lxml")
body_dirty = bsc.find(attrs={"itemprop":"articleBody"})

#remove commercials and links to different articles
for div in body_dirty.find_all("div"):
    div.decompose()

#remove some lists
for div in body_dirty.find_all("ul", attrs={"class":"inline-pipes-list"}):
    div.decompose()
body_dirty.find("a", attrs={"class":"syndication-btn"}).decompose()
body_clean = body_dirty.get_text()

sentence_clean = sent_tokenize(body_clean, "english")
f2 = open("tmp/independent_"+sys.argv[2]+'.txt', "w+")

for sentence in sentence_clean:
        print(sentence.strip())
        f2.write(sentence.strip()+"\r\n")
