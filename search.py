import csv
import sys

searchstring = sys.argv[3]
searchlist = searchstring.split(" ")

f1 = open("./scraped/"+sys.argv[1]+'.csv', "r")
f2 = open("./scraped/"+sys.argv[2]+'.csv', "r")
links = []

file1 = csv.reader(f1)
file2 = csv.reader(f2)

for r in file1:
    pres = [0] * len(searchlist)
    for c in r:
        if c.find(" "):
            itr = 0
            for t in searchlist:
                pres[itr] = c.find(t)
                itr = itr+1
        if not all(pres):
            print(r)

for r in file2:
    pres = [0] * len(searchlist)
    for c in r:
        if c.find(" "):
            itr = 0
            for t in searchlist:
                pres[itr] = c.find(t)
                itr = itr+1
        if not any(pres):
            print(r)

