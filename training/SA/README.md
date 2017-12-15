These scripts have to be started from the root of the project.

The simple sentiment analysis script dismantles the data-set into data, creates a vocabulary of the used words, pads the sequences be the longest sequence, which, in our case is a sentence. To run the script, one needs to be on the project root and run:<br />
`python training/SA/sa.py`

The Hyper-param adjustment testing script: <br />
`python training/SA/sa_hpa.py`<br />
need to be started from the document root, but has different prerequisites, compared to the rest of the project. Namely 'networkx' python library version **1.11** has to be installed instead of **2.0** which was pulled as a dependency of scrapy. So for hyper-parameter testing one needs to reinstall the requested library.

The next script, run as:
 `python training/SA/sa_w2v.py`<br />
 is similar to the first one, but the word embeddings are pre-trained vectors from w2v, for better results. Currently the difference is 5%.  