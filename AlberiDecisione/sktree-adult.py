#!/usr/bin/env python2

import pandas as pd
import numpy as np
from sklearn import tree,ensemble
from graphviz import Digraph

names=[]
with open('adult.names','r') as f:
	for line in f:
		if line[0] == '|':
			continue
		l= line.split(':') ##splitta sulla base dei : salva in l[0] la prima parte l[1] la seconda
		if len(l)>1:
			names.append(str(l[0]))

names.append('income')


adult = pd.read_csv('adult.csv', header=None,names=names,skipinitialspace=True)


m,n = adult.shape


X = pd.DataFrame()##lo riepo dando ad ogni colonna un nome significativo

for j in adult:
	col = adult[j]
	print col.name,col.dtype
	if col.dtype == np.int64:
		X[col.name] = col
	elif j != names[-1]:
		l = list(set(col)) ##valori diversi che la colonna puo assumere
		c = np.zeros((m,len(l)))
		for i in range(m):
			c[i][l.index(col[i])] = 1.0
		for k in range(len(l)):
			X[col.name+'='+l[k]] = c[:,k]




Y = adult[names[-1]]




t = tree.DecisionTreeClassifier(max_depth=3)
t.fit(X, Y)
tree.export_graphviz(t, 'adult.dot')

X_names = X.columns.values

with open('adult.dot','r') as f, open('adult2.dot','w') as g:
	for line in f:
		p=line.find('X[')
		if p>=0:
			q=line.find(']')
			n=int(line[p+2])
			line = line[:p] + X_names[n] + line[q+1:]
		g.write(line)



inx=range(m)
np.random.shuffle(inx)
m_tr = int(m*0.75)

X_val = X[inx[m_tr:]]
Y_val = Y[inx[m_tr:]]
X_tr = X.iloc[inx[:m_tr]]
Y_tr = Y.iloc[inx[:m_tr]]

t2 = tree.DecisionTreeClassifier()
t2.fit(X_tr, Y_tr)

Ypred = t2.predict(X_val)

def confusionMatrix (Y,Yt):
	#caso positivo se >, negativo se <=
	c = np.zeros((2,2),dtype=int)
	for y,yt in zip(Y,Yt):
		iv = 1 if y[0] == '>' else 0
		ip = 1 if yt[0] == '>' else 0 
		c[iv,ip] +=1

	return c

print confusionMatrix(Y_val,Ypred)



t3 = ensemble.RandomForsetClassifier()
t3.fit(X_tr,Y_tr)
Ypred3 = t3.predict(X_val)

print confusionMatrix(Y_val,Ypred3)

for n_estimators in [10,100,1000]:
	t4= ensable.RandomForsetClassifier(n_estimators = n_estimators,max_depth=5,max_features =2,class_weight ='balanced')
	t4.fit(X_tr,Y_tr)
	Ypred4 = t4.predict(X_val)
	print n_estimators
	print confusionMatrix(Y_val,Ypred4)






