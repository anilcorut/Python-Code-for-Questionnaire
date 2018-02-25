import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection
#models to use
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

#model evaluation approaches
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


#database handling
from bs4 import BeautifulSoup
import os
import urllib2
import re
import sqlite3 as lite
import json

dbName="fragrances.sqlite"
connection = lite.connect(dbName)
connection.text_factory = str
df = pd.read_sql_query("select * from Fragrances;", connection)
raw = df

cursor=connection.cursor()

notes=cursor.execute("select notes from Fragrances;").fetchall()
#notes=cursor.fetchall()
unqNotes=[]
for note in notes:
    for each in note[0].split(','):
        unqNotes.append(each.strip())

unqNotes=list(set(unqNotes))

#for each unique note we created a coloumn and filled with 0 value, it is a boolean variable, if that particular fragnance has a this note in it, then this value will become 1 later.
for each in unqNotes:
    df[each] = 0

#new coloumns are generated and filled with 0.0 values
df['mainNote']=''
df['sillageSoft']=0.0
df['sillageModerate']=0.0
df['sillageHeavy']=0.0
df['sillageEnormous']=0.0
df['longevityPoor']=0.0
df['longevityWeak']=0.0
df['longevityModerate']=0.0
df['longevityLongL']=0.0
df['longevityVeryLongL']=0.0

#\n characters cleared in the end of sillage values
for each in df['sillage']:
    each=each[:len(each)-1]

for index, row in df.iterrows():
    row['mainNote']=row['notes'].split(',')[0].strip()
    df.set_value(index, 'mainNote', row['mainNote'])
    for note in unqNotes:
        for eachIteration in row['notes'].split(','):
            if eachIteration.strip()==note:
                df.set_value(index, note, 1)

    sumOfSillage=0
    #the sillage value is currently something like 30,50,60,70 but we need proportion of them
    #so first we sum all of these values
    for sillages in row['sillage'].split(','):
        print sillages,
        sumOfSillage+=int(sillages.strip())
    sumOfSillage=float(sumOfSillage)

    sumOfLongevity = 0
    for longevities in row['longevity'].split(','):
        print longevities,
        sumOfLongevity += int(longevities.strip())
    sumOfLongevity=float(sumOfLongevity)
    #we divide this sillage coloumn into 4 different coloumns and set their values as each/sum
    
    if sumOfSillage>0:
        row['sillageSoft'] = row['sillage'].split(',')[0].strip()
        df.set_value(index, 'sillageSoft', '{0:.2f}'.format(int(row['sillageSoft']) / sumOfSillage))
        row['sillageModerate'] = row['sillage'].split(',')[1].strip()
        df.set_value(index, 'sillageModerate', '{0:.2f}'.format(int(row['sillageModerate']) / sumOfSillage))
        row['sillageHeavy'] = row['sillage'].split(',')[2].strip()
        df.set_value(index, 'sillageHeavy', '{0:.2f}'.format(int(row['sillageHeavy']) / sumOfSillage))
        row['sillageEnormous'] = row['sillage'].split(',')[3].strip()
        df.set_value(index, 'sillageEnormous', '{0:.2f}'.format(int(row['sillageEnormous']) / sumOfSillage))
        row['longevityPoor'] = row['longevity'].split(',')[0].strip()
    #if all the values are 0 this process is avoided and all the coloumns are filled with 0
    else:
        row['sillageSoft'] = row['sillage'].split(',')[0].strip()
        df.set_value(index, 'sillageSoft', row['sillageSoft'])
        row['sillageModerate'] = row['sillage'].split(',')[1].strip()
        df.set_value(index, 'sillageModerate', row['sillageModerate'])
        row['sillageHeavy'] = row['sillage'].split(',')[2].strip()
        df.set_value(index, 'sillageHeavy', row['sillageHeavy'])
        row['sillageEnormous'] = row['sillage'].split(',')[3].strip()
        df.set_value(index, 'sillageEnormous', row['sillageEnormous'])
    #same process is done for longevity values
    if sumOfLongevity>0:
        row['longevityPoor'] = row['longevity'].split(',')[0].strip()
        df.set_value(index, 'longevityPoor', '{0:.2f}'.format(int(row['longevityPoor'])/sumOfLongevity))
        row['longevityWeak'] = row['longevity'].split(',')[1].strip()
        df.set_value(index, 'longevityWeak', '{0:.2f}'.format(int(row['longevityWeak']) / sumOfLongevity))
        row['longevityModerate'] = row['longevity'].split(',')[2].strip()
        df.set_value(index, 'longevityModerate', '{0:.2f}'.format(int(row['longevityModerate']) / sumOfLongevity))
        row['longevityLongL'] = row['longevity'].split(',')[3].strip()
        df.set_value(index, 'longevityLongL', '{0:.2f}'.format(int(row['longevityLongL']) / sumOfLongevity))
        row['longevityVeryLongL'] = row['longevity'].split(',')[4].strip()
        df.set_value(index, 'longevityVeryLongL', '{0:.2f}'.format(int(row['longevityVeryLongL']) / sumOfLongevity))

    else:
        row['longevityPoor'] = row['longevity'].split(',')[0].strip()
        df.set_value(index, 'longevityPoor', row['longevityPoor'])
        row['longevityWeak'] = row['longevity'].split(',')[1].strip()
        df.set_value(index, 'longevityWeak', row['longevityWeak'])
        row['longevityModerate'] = row['longevity'].split(',')[2].strip()
        df.set_value(index, 'longevityModerate', row['longevityModerate'])
        row['longevityLongL'] = row['longevity'].split(',')[3].strip()
        df.set_value(index, 'longevityLongL', row['longevityLongL'])
        row['longevityVeryLongL'] = row['longevity'].split(',')[4].strip()
        df.set_value(index, 'longevityVeryLongL', row['longevityVeryLongL'])



connection.close()

#cleaning gender column
df['gender'].replace('women and men', 'unisex',inplace=True)
df['gender'].replace('women ', 'women',inplace=True)
df['gender'].replace('men salvador dali cologne', 'men',inplace=True)
df['gender'].replace('men  givenchy cologne', 'men',inplace=True)
df['gender'].replace('men ', 'men',inplace=True)
df = df.drop(df[df.gender == '3, 0, 0, 20, 10'].index)
df = df.drop(df[df.gender == '2'].index)

#Creating Dummy Variables based on variable name (varn)
varn = 'gender'
dummies = pd.get_dummies(df[varn]).rename(columns=lambda x: varn+ "_" + str(x))
df = pd.concat([df, dummies], axis=1)
del df[varn]

#removing other junk
del df['id']

#rating distribution
import matplotlib.pyplot as plt
plt.hist(y)
plt.title("Gaussian Histogram for Rating")
plt.xlabel("Rating")
plt.ylabel("Frequency")

fig = plt.gcf()
fig

# changing continous rating to binary classification 1 and 0
for index, row in df.iterrows():
    if row['rating'] >= 4.0:
        df.set_value(index=index, col='rating', value='1.0')
    else:
        df.set_value(index=index, col='rating', value='0')
        
#dropping these variables to only keep numeric and dummy variables
l = ['name','brand','groupName','love','like','dislike','winter','spring','summer','autumn','day','night','rating_count','notes','longevity','sillage','mainNote','sillageSoft','sillageModerate','sillageHeavy','sillageEnormous','longevityPoor','longevityWeak','longevityModerate','longevityLongL','longevityVeryLongL']
data =df
#choosing subset of variables
num_feats = list(set(data.columns) - set(l))
#seperating training input and training output y
X = data[num_feats]
y = data['rating']



#results using kfold cross validation and box plots
#seed to keep randomisation same to replicate
seed = 7
# prepare models
models = []
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LogR', LogisticRegression()))




#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('SVM', SVC()))
models.append(('ANN', MLPClassifier()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


