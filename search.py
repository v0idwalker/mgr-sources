import csv
import sys
import string

searchstring = sys.argv[3]
# print(searchstring)
searchlist = searchstring.split(" ")


# print(sys.argv[1] + " " + sys.argv[2])
f1 = open("./scraped/"+sys.argv[1]+'.csv', "r")
f2 = open("./scraped/"+sys.argv[2]+'.csv', "r")
links = []

file1 = csv.reader(f1)
file2 = csv.reader(f2)

for r in file1:
    # print(searchlist)
    pres = [0] * len(searchlist)
    # print(pres)
    # print("NOW")
    for c in r:
        if c.find(" "):
            itr = 0
            for t in searchlist:
                pres[itr] = c.find(t)
                itr = itr+1
        if not all(pres):
            print(r)

for r in file2:
    pass

# print(searchfor)
# file1
