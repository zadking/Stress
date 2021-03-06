import csv
from sklearn.feature_selection import VarianceThreshold
import numpy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

def findlabel(part,act,data):
    index = 0
    for val in data:
        if val[0] in part:
            if val[1] in act:
                return index
        index += 1

def cohens(adata,bdata,labela,labelb,iterations):
    cohens = {}
    for trua in range(iterations):
        trutha = trua + 1
        x = {}
        for trub in range(iterations):
            truthb = trub + 1
            ia = 0
            a = 0
            b = 0
            c = 0
            d = 0
            for val in labela:
                ib = findlabel(val[0],val[1],labelb)
                if adata[ia] == trutha:
                    #print b[ib]
                    if bdata[ib] == truthb:
                        a += 1
                    else:
                        b += 1
                else:
                    #print b[ib]
                    if bdata[ib] == truthb:
                        c += 1
                    else:
                        d += 1
                ia += 1
            total = a+b+c+d
            probo = (a + d) / float(total)
            probyes = ((a + b)/float(total)) *((a + c) / float(total))
            probno = ((c + d)/float(total)) *((b + d) / float(total))

            x.update({truthb:(probo - (probyes + probno))/(1-(probyes + probno))})
        cohens.update({trutha:x})
    return cohens

    
with open("labelsfeatures.csv", "rb") as flabels:
    flabels = list(csv.reader(flabels,delimiter=","))
    flabels = flabels[2:]

with open("srlabel.csv", "rb") as srlabels:
    srlabels = list(csv.reader(srlabels,delimiter=","))
fclusters = pd.read_csv('featureselectionlabels.csv')
srcluster = pd.read_csv('srclusters.csv')
for fcol in fclusters:
    for scol in srcluster:
        print [fcol,scol]
        print cohens(list(fclusters[fcol]),list(srcluster[scol]),flabels,srlabels,2)



