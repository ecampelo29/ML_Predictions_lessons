#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:45:21 2020

@author: ecampelo
"""

# importações necessárias
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import nltk
from collections import Counter
import numpy as np

# funções pessoais (incluir local no pythonpath)
import ML_functions as func

# settings
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# variáveis           
# download_files = os.path.join(os.path.expanduser('~'), 'Documentos', 
 #                                 'Machine learning', 'Data_files')

# spam_files = os.path.join(download_files, 'spam')
# ham_files =  os.path.join(download_files, 'easy_ham')


from sklearn import datasets

iris = datasets.load_iris()

# print(iris['DESCR'])


X = iris['data']

X_petal = iris['data'][:, (2, 3)]  # petal length, petal width
X_sepal = iris['data'][:, (0, 1)] # setal length and width


y = iris["target"]


relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                 'Machine learning', 'Resultados_ML_Iris.txt')

# =============================================================================
# objetos
# =============================================================================

# =============================================================================
# analisando os dados 
# =============================================================================

# pétalas
plt.figure(figsize=(12, 6))
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
# plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
# plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Sepal length", fontsize=14)
plt.ylabel("Sepal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 9, 0, 5])
plt.show()


# separando os dados - primeiro a parte linear. 
X_train, X_test, y_train, y_test = train_test_split(X_petal, y, test_size= 0.2, random_state=42)

# transformando os dados 
from sklearn.preprocessing import StandardScaler
st_scaler = StandardScaler()

# X_train_scaled = st_scaler.fit_transform(X_train)
# X_test_scaled = st_scaler.fit_transform(X_test)

# =============================================================================
# Linear Regression
# =============================================================================
from sklearn.linear_model import LogisticRegression

lr_clf_petal = LogisticRegression()

# lr_clf_petal.fit(X_train_scaled, y_train)
# y_pred = lr_clf_petal.predict(X_test)

# y_scores = lr_clf_petal.decision_function(X_test)

# metricas = func.classification_metrics(relatorio, lr_clf_petal, y_test, y_pred, y_scores)

# print(metricas) # 

# resultado: não funciona para mais de uma classe


# transformando então y em binarios, adaptando as análise para multiclass

y_setosa = (y==0).astype(int)
y_versicolor = (y==1).astype(int)
y_virginica = (y==2).astype(int)

score_list = []
y_list = []

for flower in [y_setosa, y_versicolor,y_virginica]:
    X_train, X_test, y_train, y_test = train_test_split(X_petal, flower, test_size= 0.2, random_state=42)
    y_list.append(y_test)
    X_train_scaled = st_scaler.fit_transform(X_train)
    X_test_scaled = st_scaler.fit_transform(X_test)
    lr_clf_petal.fit(X_train_scaled, y_train)
    
    # y_pred = lr_clf_petal.predict(X_test_scaled)

    y_scores = lr_clf_petal.decision_function(X_test_scaled)
    score_list.append(y_scores)
    

    # print(flower,'\n', metricas)
y_pred= []
y_scores_final = []
for score in zip(score_list[0],score_list[1],score_list[2]):
    y_scores_final.append(max(score))
    y_pred.append(score.index(max(score)))

# transformando em array para gerar as métricas
y_pred = np.asarray(y_pred)
y_scores_final = np.asarray(y_scores_final)

# criando o y_test para análise dos resultados preditos
# virginica 
y_list[2][y_list[2] == 1] = 2
# a partir do y_list[1] - versicolor, somando com o y_list[2] virginica temos todas as flores
y_test_final = y_list[2]+y_list[1]


metricas = func.classification_metrics(relatorio, lr_clf_petal, y_test_final, y_pred, y_scores_final)

print(metricas)
###########################
# 100% de acerto no teste (overfitting?!?!?)

# =============================================================================
# Usando modelos próprios para multiclass
# =============================================================================








