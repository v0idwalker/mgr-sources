#!/usr/bin/env bash
# run scapy to scape links
# select the links containing similar tags/words
# run ner/sa
# print out results

SEARCH=$1
PWD=`pwd`
TIME=`date +"%s"`
CRAWL=./crawl/

echo "Scraping sites"

    cd $CRAWL
    NAME1="$TIME"-independet
#    echo $NAME1
    scrapy runspider ./crawl/spiders/independent.py --output ../scraped/"$NAME1".csv 2>&1 | tee -a ../scraped/log-"$TIME".log
    NAME2="$TIME"-theguardian
#    echo $NAME2
    scrapy runspider ./crawl/spiders/theguardian.py --output ../scraped/"$NAME2".csv 2>&1 | tee -a ../scraped/log-"$TIME".log
    cd ..

echo "Searching strings $SEARCH"

python ./search.py "$NAME1" "$NAME2" "$SEARCH"

echo "Which do you wish to compare from $NAME1?"
read WHICH1
echo "Which do you wish to compare it to, fom $NAME2?"
read WHICH2

# contains the guessing sequence by NN
