# -*- coding: utf-8 -*-
# fetch data
import requests
from bs4 import BeautifulSoup
import sys


website = sys.argv[1]
body_tags = sys.argv[2]

def content_body(body_tags):
    r = requests.get("url")
    bsc = BeautifulSoup(r.content)
    body_dirty = bsc.find(body_tags)

    # remove HTML tags, get only the text.
    body_clean = body_dirty
    return body_clean

