# pre-process data, by removing stuff that does not help store in ES
import requests
from bs4 import BeautifulSoup as BS

urls = [] #from spiders

for url in urls:
    r = requests.get(url)
    soup = BS(r.content)
    HText = soup.get()
    # now clean from useless tags
