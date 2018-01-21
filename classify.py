import pandas as pd
import csv
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,MiniBatchKMeans,SpectralClustering
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit

def findlabel(part,act,data,definitions):
    index = 0
    for val in data:
        if val[0] == part:
            if val[1] in act:
                return definitions[index]
        index += 1
    print [part,act]


ttest = pd.read_csv('ttestfeaturedata.csv')
act = ttest['activity']
index =0
TSST = []
Rest = []
features = []
for val in act:
    if 'Arithmetic' in val or 'Speech' in val:
        TSST.append(index)
    if 'rest' in val:
        Rest.append(index)
    index += 1
for col in ttest:
    if 'Participant' in col or 'activity' in col or 'window_number' in col:
        continue
    else:
        x = ttest[col].iloc[TSST]
        y = ttest[col].iloc[Rest]
        stat, p =  stats.ttest_ind(list(x),list(y))
        if p < .05:
            features.append(col)
ecgfeatures = [x for x in features if 'ecg' in x]
gsrfeatures = [x for x in features if 'neulog' in x]
ecggsrfeatures = ecgfeatures + gsrfeatures
df = pd.read_csv('definition.csv')
dm = df[['part','activity']]
da = dm.as_matrix()
df = df[['yesNo','average','pss','consensus']]
part_label = list(ttest['Participant'])
act_label = list(ttest['activity'])
for i in range(0,3):
    
    if i == 0:
        name = 'ecg'
        cluster = ecgfeatures
    elif i == 1:
        name = 'gsr'
        cluster = gsrfeatures
    elif i == 2:
        name = 'ecg+gsr'
        cluster = ecggsrfeatures
    with open(name+'generalized.csv','w') as csvfile:
        write = csv.writer(csvfile,delimiter= ',')
        with open(name+'personalized.csv','w') as pcsvfile:
            pwrite = csv.writer(pcsvfile,delimiter= ',')
            for col in df:
                definitions = list(df[col])
                deflabelmin = []
                for i in range(len(ttest.as_matrix())):
                    deflabelmin.append(findlabel(part_label[i],act_label[i],da,definitions))
                    labels = pd.DataFrame(data = deflabelmin, columns = ['Label'])
                    totaldata = pd.concat([ttest,labels], axis = 1)
            
                for p in np.unique(part_label):
                    print p
                    testtotal = totaldata.loc[totaldata['Participant'] == p]
                    testdata = testtotal[cluster].as_matrix()
                    testlabel = testtotal['Label'].as_matrix()
                    ptestdata = testtotal[cluster]
                    ptestlabel = testtotal['Label']
                    
                    
                   
                    traintotal = totaldata.loc[totaldata['Participant'] != p]
                    traindata = traintotal[cluster].as_matrix()
                    trainlabel = traintotal['Label'].as_matrix()
                    clf = SVC()                    
                    pclf = SVC()
                    groups = testtotal['activity'].as_matrix()
                    
                    ptestdata = testtotal.loc[totaldata['activity'] == a][cluster].as_matrix()
                    ptestlabel = testtotal.loc[totaldata['activity'] == a]['Label'].as_matrix()
                    ptraindata = testtotal.loc[totaldata['activity'] != a][cluster].as_matrix()
                    ptrainlabel = testtotal.loc[totaldata['activity'] != a]['Label'].as_matrix()
                    gss = GroupShuffleSplit(n_splits=2, test_size=0.5, random_state=0)
                    Xtrain, Xtest = gss.split(ptestdata.index, ptestlabel,  groups = groups)
                    X_train = ptestdata.iloc[Xtrain[0]].as_matrix()
                    X_test = ptestdata.iloc[Xtest[1]].as_matrix()
                    y_train = ptestlabel.iloc[Xtrain[0]].as_matrix()
                    y_test = ptestlabel.iloc[Xtest[1]].as_matrix()
                    if len(np.unique(y_train)) == 1 or len(np.unique(y_test)) == 1:
                        pf1 = 'Groundtruth is all set to 0'
                    else:
                        pclf.fit(ptraindata,ptrainlabel)
                        ppredicted = pclf.predict(ptestdata)
                        pf1 = f1_score(ptestlabel,ppredicted)
                        pwrite.writerow([name,p,col,pf1,len(np.unique(y_train)),len(np.unique(y_test))])
                    if len(np.unique(testlabel)) == 1 or len(np.unique(trainlabel)) == 1:
                        f1 = 'Groundtruth is all set to 0'
                    else:
                        clf.fit(traindata,trainlabel)
                        predicted = clf.predict(testdata)
                        f1 = f1_score(testlabel,predicted)
                            
                    
                    write.writerow([name,p,col,f1])
       
                          
    
