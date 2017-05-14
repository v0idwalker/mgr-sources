# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import sys

# windows specific requirenment for printing out
import win_unicode_console
win_unicode_console.enable()

website = sys.argv[1]

r = requests.get(website)
bsc = BeautifulSoup(r.content, "lxml")
body_dirty = bsc.find("div", attrs={"class": "content__article-body", "itemprop": "articleBody"})

#remove commercials and links to different articles
for div in body_dirty.find_all("figure"):
    div.decompose()

for div in body_dirty.find_all("aside"):
    div.decompose()

for div in body_dirty.find_all("div", attrs={"class": "js-ad-slot"}):
    div.decompose()

body_clean = body_dirty.get_text()
print(body_clean)


