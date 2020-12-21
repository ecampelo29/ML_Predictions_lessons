#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 09:45:21 2020

@author: ecampelo
"""

# importações necessárias
import os 
import matplotlib as mpl

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
download_files = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'Data_files')

spam_files = os.path.join(download_files, 'spam')
ham_files =  os.path.join(download_files, 'easy_ham')


relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                 'Machine learning', 'Resultados_ML_Spam.txt')

# =============================================================================
# objetos
# =============================================================================


# adaptando o objeto do livro 
# class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.stemming = True
       
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         stemmer = nltk.PorterStemmer()
#         X_transformed = []
#         for msg in list(X['body']):
#             word_counts = Counter(msg.split())
#             if self.stemming and stemmer is not None:
#                 stemmed_word_counts = Counter()
#                 for word, count in word_counts.items():
#                     stemmed_word = stemmer.stem(word)
#                     stemmed_word_counts[stemmed_word] += count
#                 word_counts = stemmed_word_counts
#             X_transformed.append(word_counts)
#         return np.array(X_transformed)

# incluindo o assunto para ver se melhora o desempenho no hard ham
class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemming = True
       
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        stemmer = nltk.PorterStemmer()
        X_transformed = []
        for subject, msg in zip(list(X['subject']), list(X['body'])):
            sub_word_count = Counter(subject.split())
            word_counts = Counter(msg.split())
            if self.stemming and stemmer is not None:
                stemmed_subword_counts = Counter()
                stemmed_word_counts = Counter()
                for word, count in sub_word_count.items():
                    stemmed_subject = stemmer.stem(word)
                    stemmed_subword_counts[stemmed_subject] += count
                sub_word_count = stemmed_subword_counts
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append((sub_word_count, word_counts))
        return np.array(X_transformed)






# class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, vocabulary_size=1000):
#         self.vocabulary_size = vocabulary_size
#     def fit(self, X, y=None):
#         total_count = Counter()
#         for word_count in X:
#             for word, count in word_count.items():
#                 total_count[word] += min(count, 10)
#         most_common = total_count.most_common()[:self.vocabulary_size]
#         self.most_common_ = most_common
#         self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
#         return self
#     def transform(self, X, y=None):
#         rows = []
#         cols = []
#         data = []
#         for row, word_count in enumerate(X):
#             print(word_count)
#             for word, count in word_count.items():
#                 rows.append(row)
#                 cols.append(self.vocabulary_.get(word, 0))
#                 data.append(count)
                
#         return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
    

# ajustando para considerar transformação do assunto do email
class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for subject, body in X:
            for word, count in subject.items():
                total_count[word] += min(count, 10)
                most_common = total_count.most_common()[:self.vocabulary_size]
                self.most_common_ = most_common
                self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
            for word, count in body.items():
                total_count[word] += min(count, 10)
                most_common = total_count.most_common()[:self.vocabulary_size]
                self.most_common_ = most_common
                self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    
    def transform(self, X, y=None):
        row = -1
        rows = []
        cols = []
        data = []
        for subject, body in X:
            row +=1
            for word, count in subject.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
            sub_matrix = csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
            for word, count in body.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
            body_matrix = csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
            
                
        return csr_matrix(np.concatenate((sub_matrix.toarray(), body_matrix.toarray()), axis=1))
            
 


# =============================================================================
# 
# =============================================================================
# few = X_train[:3]

# few_t = EmailToWordCounterTransformer().fit_transform(few)

# few_tra = WordCounterToVectorTransformer().fit_transform(few_t)

# few_tra = WordCounterToVectorTransformer().fit_transform(few_t)


# =============================================================================
# analisando alguns arquivos spam
# =============================================================================
# import re
# import os
# import urlextract


# file =[
# '00001.7848dde101aa985090474a91ec93fcf0',
# '00022.8203cdf03888f656dc0381701148f73d',
# '00050.45de99e8c120fddafe7c89fb3de1c14f', 
# '00135.00e388e3b23df6278a8845047ca25160', 
# '00164.8536500ed9cadc8397a63b697d043c0b'
# ]
# mail = []

# for msg in file:
#      with open(os.path.join(spam_files, msg ), "rb") as f:
#          e = email.parser.BytesParser(policy=email.policy.default).parse(f) 
#      # alguns email tem multpart - precisamos apenas do texto
#      for part in e.walk():
#          if part.get_content_type() in ('text/plain', 'text/html'):
#              try:
#                  e_limpo = limpa_html(part.get_content())
#              except LookupError: # encoding erro
#                  e_limpo = part.get_payload(decode=True)
#                  e_limpo = e_limpo.decode('windows-1252')
#      mail.append(e_limpo)
                    
# print(e.get_content().strip())
# # mesmo spam são varias estruturas/emails em um só arquivo.
# for header, values in e.items():
#     print(header, ':', values)
    
# e.values()
# e.keys()
# # o que interessa por enquanto
# e['Subject'] # assunto
# e['From'] # emitente
# # corpo do email
# corpo = e.get_content() # precisa de limpeza para tirar o html  e tratamento multipart

# =============================================================================
# Carregamento das mensagens em dataframes
# =============================================================================
# spams 547
df_spam = func.email_to_dataframe(spam_files, 1)
# não spams 2522
df_not_spam = func.email_to_dataframe(ham_files)

# universo
df_all_emails = df_spam.append(df_not_spam, ignore_index=True)

# analisando disposição dos dados
df_desc = df_all_emails.describe() # temos 2 mensagens com NA
df_all_emails.dropna(inplace = True) # eliminando não irá afetar o estudo

# liberando memória
df_spam = df_not_spam = []
# =============================================================================
# antes de seguir, separar dos dados para treinamento
# =============================================================================

df_train, df_test  = train_test_split(df_all_emails, test_size=0.2, 
                                    random_state=42)

X_train = df_train.drop('spam', axis=1).copy()
y_train = df_train['spam'].copy().astype('int')
X_test  = df_test.drop('spam', axis=1).copy()
y_test =  df_test['spam'].copy().astype('int')


# =============================================================================
# Stemming 
# =============================================================================

# df = X_train[:5]

# X_df = EmailToWordCounterTransformer().fit_transform(df)

# vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
# X_df_vectors = vocab_transformer.fit_transform(X_df)
# X_df_vectors.toarray()

# =============================================================================
# Pipeline que retorna os textos (body) em matriz
# =============================================================================

# preprocess_pipeline=[]
# preprocess_pipeline = Pipeline([
#     ("email_to_wordcount", EmailToWordCounterTransformer()),
#     ("wordcount_to_vector", WordCounterToVectorTransformer()),
# ])

# X_train_transformed = preprocess_pipeline.fit_transform(X_train)
# X_test_transformed = preprocess_pipeline.transform(X_test)

# pipeline passou a apresentar erro no dicionário ??  se processado um por vez
# funciona normalmente... 
X_e = EmailToWordCounterTransformer().fit_transform(X_test)
X_test_transformed = WordCounterToVectorTransformer().fit_transform(X_e)

X_e = EmailToWordCounterTransformer().fit_transform(X_train)
X_train_transformed = WordCounterToVectorTransformer().fit_transform(X_e)
# =============================================================================
# Treino
# =============================================================================

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(random_state=42, max_iter=1000)

lr_clf.fit(X_train_transformed, y_train)

y_pred = lr_clf.predict(X_test_transformed)

y_scores = lr_clf.decision_function(X_test_transformed)

metricas = func.classification_metrics(relatorio, lr_clf, y_test, y_pred, y_scores)

print(metricas)

# para eventual análise dos dados, junta as predições ao dataframe
df_email_pred = X_test.copy()
df_email_pred['spam'] = y_test
df_email_pred['lr_clf'] = y_pred.tolist()


# -------------------------
from sklearn.neighbors import KNeighborsClassifier

kn_clf = KNeighborsClassifier()

kn_clf.fit(X_train_transformed, y_train)

y_pred = kn_clf.predict(X_test_transformed)

y_probas = kn_clf.predict_proba(X_test_transformed)
y_scores = y_probas[:,1]

metricas = func.classification_metrics(relatorio, kn_clf, y_test, y_pred, y_scores)

print(metricas) # pouco pior que logistic

# =============================================================================
# avaliando desempenho com hard_ham 
# =============================================================================
df_all_emails = []

new_emails = os.path.join(download_files, 'hard_ham')
df_new_emails = func.email_to_dataframe(new_emails)

df_new_emails.dropna(inplace = True) 

df_label = df_new_emails['spam'].astype('int')
df_new_emails.drop('spam', axis=1, inplace=True)


# new_data_transformed = preprocess_pipeline.transform(df_new_emails)
X_e = EmailToWordCounterTransformer().fit_transform(df_new_emails)

new_data_transformed = WordCounterToVectorTransformer().fit_transform(X_e)
# muito ruim!!! 48% // incluindo análise do assunto 88% de acerto
func.predicting(lr_clf,new_data_transformed, df_label, relatorio)

# um pouco melhor 87% // incluindo análise do assunto = 95% de acerto
func.predicting(kn_clf,new_data_transformed, df_label, relatorio)


# muito ruim!!! 46% // 69%
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)    
sgd_clf.fit(X_train_transformed, y_train)

func.predicting(sgd_clf, new_data_transformed, df_label, relatorio)

# muito ruim!!! 37% // 79%
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42)

dt_clf.fit(X_train_transformed, y_train)

func.predicting(dt_clf, new_data_transformed, df_label, relatorio)

# melhor... 64% /// 98% - passa a ser o melhor modelo.
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train_transformed, y_train)

func.predicting(forest_clf, new_data_transformed, df_label, relatorio)


y_pred = lr_clf.predict(X_test_transformed)

y_scores = lr_clf.decision_function(X_test_transformed)

metricas = func.classification_metrics(relatorio, lr_clf, y_test, y_pred, y_scores)

print(metricas)



# melhor... 84% // 94%
from sklearn.svm import SVC
svm_clf = SVC(gamma="scale",  probability=True, random_state=42)

svm_clf.fit(X_train_transformed, y_train)

func.predicting(svm_clf, new_data_transformed, df_label, relatorio)

# =============================================================================
# Ajustando o melhor modelo até agora: KNN
# =============================================================================

# K - quantidade de vizinhos a serem usados no calculo da distância 
ks = list(range(3,31))

# tipo de cálculo da distância 1 = Manhattan 2 = Euclidian
ps = [1,2]

# leaf_size influencia o algorítimo por traz dos calculos
leaves = list(range(1,101))

hyper_tuning = dict(n_neighbors=ks, leaf_size=leaves, p=ps)

# zerando o modelo
kn_clf = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(kn_clf, hyper_tuning, cv=10)

gs_clf = gs.fit(X_train_transformed, y_train)

best_params = gs_clf.best_params_

# pior que o default 81% ??
best_kn_clf = gs_clf.best_estimator_

func.keep_model(best_kn_clf, 'best_kn_clf_spam')

best_kn_clf = func.load_model('best_kn_clf_spam')

func.predicting(best_kn_clf, new_data_transformed, df_label, relatorio)

# =============================================================================
# após incluir o assunto como variável, pois não há muitos dados para treinar
# o modelo de forma mais completa, verifica-se que no dataset de treino e teste
# as métricas não são promissoras, porém há melhora em todos os modelos
# qdo aplicado no dataset Hard Ham, chegando a 98% de identificação de não spam 
# =============================================================================






