# fetch data
import requests
from bs4 import BeautifulSoup

def content_body():
    r = requests.get("url")
    bsc = BeautifulSoup(r.content)
    bsc.find("")
    # remove HTML tags, get only the text.