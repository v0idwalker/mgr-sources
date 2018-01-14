import csv
import sys

try:
    import win_unicode_console
    win_unicode_console.enable()
except ImportError as e:
    print(e)

searchstring = sys.argv[3]
searchlist = searchstring.split(" ")

f1 = open("./scraped/"+sys.argv[1]+'.csv', "r")
f2 = open("./scraped/"+sys.argv[2]+'.csv', "r")
links = []

file1 = csv.reader(f1)
file2 = csv.reader(f2)

for r in file1:
    pres = [0] * len(searchlist)
    # for c in r:
    if (r[0].find(" ")!=-1):
        itr = 0
        for t in searchlist:
            if (r[0].lower().find(t) != -1):
                pres[itr] = 1
            itr = itr + 1
    if all(pres):
            print(r[1])
            print(pres)
            # print(searchlist)
        # print(pres)

for r in file2:
    pres = [0] * len(searchlist)
    # for c in r:
    if (r[0].find(" ")!=-1):
        itr = 0
        for t in searchlist:
            if (r[0].lower().find(t) != -1):
                pres[itr] = 1
            itr = itr+1
    if all(pres):
        print(r[1])
        print(pres)
        # print(searchlist)
    # print(pres+" "+c)
