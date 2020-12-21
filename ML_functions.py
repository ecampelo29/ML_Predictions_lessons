#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:12:20 2020

@author: ecampelo
"""
import re  
import os
import pandas as pd
import email
import email.policy
import urlextract
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score   
import joblib   

def limpa_html(msg):
    # extrator de urls
    url_extractor = urlextract.URLExtract()
    # substituindo as tags que tem apenas letras maiúsculas (multiline e newline)
    msg_limpa = re.sub('</?[A-Z]*?>', '', msg, flags=re.M | re.S )
    # remove as demais que possuem letras minúsculas e outros caracteres ignorando cases
    msg_limpa = re.sub('<.*?>', '', msg_limpa, flags=re.M | re.S | re.I)
    # eliminas as linhas intermediárias
    msg_limpa = re.sub('\n', ' ', msg_limpa.strip())
    # elimina urls
    urls = list(set(url_extractor.find_urls(msg_limpa)))
    for url in urls:
        msg_limpa = msg_limpa.replace(url, " URL ")        
    # elimina números
    msg_limpa = re.sub('\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', msg_limpa)
    # elimina pontuação
    msg_limpa = re.sub('\W+', ' ', msg_limpa, flags=re.M)
   
    return msg_limpa.lower()

   
def email_to_dataframe (folder, spam=0):
    # data frame para ter os emails de forma estruturada
    columns = ['subject', 'sender', 'body', 'spam']
    df_email = pd.DataFrame(columns = columns)
    for msg in os.listdir(folder):
        with open(os.path.join(folder, msg ), "rb") as f:
            e = email.parser.BytesParser(policy=email.policy.default).parse(f) 
        # alguns email tem multpart - precisamos apenas do texto
        for part in e.walk():
            if part.get_content_type() in ('text/plain', 'text/html'):
                try:
                    e_limpo = limpa_html(part.get_content())
                except LookupError: # encoding erro
                    e_limpo = part.get_payload(decode=True)
                    e_limpo = e_limpo.decode('windows-1252')
                try:
                    # limpando título 
                    s_limpo = limpa_html(str(e['subject']))
                except TypeError: 
                    print(msg)
                    s_limpo = 'subject'
                    
                
                e_series = pd.Series([s_limpo, e['From'], e_limpo, spam],
                             index = df_email.columns) 
                df_email = df_email.append(e_series, ignore_index=True)

    return df_email



def classification_metrics(relatorio, model, y_train, y_pred, y_scores = None):
    # analisando o resultado com confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    if cm.shape == (2,2):
    # [True Negatives  False Negatives]   [realmente não são 8    são 8 e marcou como não sendo]
    # [False Positives True Positives]    [marcou como 8 e não    marcou como 8 e acertou]
        TN, FN = cm[0]
        FP, TP = cm[1]
    else: 
        TN = FN = FP = TP = 'null'
    
    cm_text = []
    for line in cm:
        cm_text.append(str(list(line)))
            
    # Quanto o modelo está correto
    precision = precision_score(y_train, y_pred, average = 'macro')
    # Qual o poder de detecção
    recall = recall_score(y_train, y_pred, average = 'macro')
    # Ponderação entre estar correta e detectar correto
    f1 = f1_score(y_train, y_pred, average = 'macro')
    if y_scores is not None: 
        try:
            if y_scores.shape[1] > 1:
                roc_score = "não disponível"
        except IndexError:    
            try:
                roc_score = roc_auc_score(y_train, y_scores)
            except ValueError:
                roc_score = "não disponível"
                


    resultado = ['Modelo: {}'.format(model), 
                 'Tue Negatives  : '+str(TN) +' False Negatives: '+str(FN),
                 'False Positives: '+str(FP)+ ' True Positives : '+str(TP),
                 'Precision Score: '+str(precision),
                 'Recall Score: '+str(recall),
                 'F1_Score: '+str(f1),
                 'Roc_auc_score: '+str(roc_score),
                 'Classes analisadas: \n'+str(list(model.classes_))+'\n',
                 'Confusion matrix: \n'
                 ]
    if os.path.isfile(relatorio):
        with open (relatorio, 'a') as r:
            r.write('\n\n'+'\n'.join(resultado))
            r.write('\n'.join(cm_text))
    else: 
        with open (relatorio, 'w') as r:
            r.write('\n\n'+'\n'.join(resultado))
            r.write('\n'.join(cm_text))
                
    return ('\n\n'+'\n'.join(resultado))


def predicting (model, transformed_data, y_label, relatorio):
    y_pred = model.predict(transformed_data)
    try:
        y_scores = model.decision_function(transformed_data)
    except AttributeError:
        y_probas = model.predict_proba(transformed_data)
        y_scores = y_probas[:,1]
    metricas = classification_metrics(relatorio, model, y_label, y_pred, y_scores)
    print(metricas)
    
# guarda o modelo para uso posterior
def keep_model (my_model, file_name):
    file = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', file_name+'.pkl')
    joblib.dump(my_model, file)
 
# carrega o modelo treinado     
def load_model (model):
    return joblib.load(os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', model+'.pkl') )
