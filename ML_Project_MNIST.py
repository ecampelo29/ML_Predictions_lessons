# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

# importações necessárias
import os 
import matplotlib as mpl
import matplotlib.pyplot as plt  
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

import joblib


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

relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                 'Machine learning', 'Resultados_ML_Mnist.txt')

# guarda o modelo para uso posterior
def keep_model (my_model, file_name):
    file = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', file_name+'.pkl')
    joblib.dump(my_model, file)
 
# carrega o modelo treinado     
def load_model (model):
    return joblib.load(os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', model+'.pkl') )

# mostra os scores (retirada do material do livro)
def display_scores(scores):    
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def classification_metrics(model, y_train, y_pred, y_scores = None):
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
                roc_score = roc_auc_score(y_train, y_scores)


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

# usando função do livro para mostrar as curvas de precision e recall
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

# função do livro para plotar a curva ROC
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

# =============================================================================
# Processando os dados
# =============================================================================

# carrengando os dados do Mnist 
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape, y.shape

plt.figure(figsize=(10,12))
# analisando conteúdo 
for d in range(10):
    some_digit = X[d]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    plt.xlabel(y[d])
    plt.show()

# cada registro possui 784 (28x28) valores de intensidade de preto, variando 
# de 0 (branco) a 255 (preto)
# este data set já está otimizado para ml, logo não há transformações especiais
# a serem feitas. Inclusive, já estão no formato de matriz

# transformando as labels em inteiro
y = y.astype(int)

# separando os dados para treinamento, validação e teste
X_train, X_val, X_test = X[:50000], X[50000:60000], X[60000:]
y_train, y_val, y_test = y[:50000], y[50000:60000], y[60000:]


# preparando as labels para análise binária, ou seja, inicialmente o modelo
# deverá classificar se o número é igual ou diferente do número 8 

# marca se a lable é ou não 8 (True/False)
y_train_oito = y_train==8 
y_val_oito = y_val == 8
y_test_oito = y_test==8

# como a intensidade para nós não parece muito importante 
# vou trabalhar com a idéia de pixel ligado (1) e pixel desligado (0), 
# comparando o resultado com o treinamento existente no livro
X_pixel = X.copy()
# troca os valores de todos maiores que zero para 1
X_pixel[X_pixel > 0] = 1


X_pixel_train, X_pixel_val, X_pixel_test = X_pixel[:50000], X_pixel[50000:60000], X_pixel[60000:]


# analisando conteúdo 
for d in range(5):
    some_digit = X_pixel[d]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap=mpl.cm.binary)
    plt.xlabel(y[d])
    plt.show()


# =============================================================================
# Treinando SGDC - Binário - Oito/não Oito
# =============================================================================
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_oito)

sgd_clf.fit(X_pixel_train, y_train_oito)

# analisando sobre os dados de validação
y_val_pred = sgd_clf.predict(X_val)
y_val_scores = sgd_clf.decision_function(X_val)



print(classification_metrics(sgd_clf, y_val_oito, y_val_pred, y_val_scores))

# gerando dados para análise das metricas
precisions, recalls, thresholds = precision_recall_curve(y_val_oito, y_val_scores)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

# outra medida é a curva ROC
fpr, tpr, trhesholds = roc_curve(y_val_oito, y_val_scores)

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.show()

# ------------- Pixel time ---ficou bem mais próximos do KNN
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_pixel_train, y_train_oito)

# analisando sobre os dados de validação
y_val_pred = sgd_clf.predict(X_pixel_val)
y_val_scores = sgd_clf.decision_function(X_pixel_val)


print(classification_metrics(sgd_clf, y_val_oito, y_val_pred, y_val_scores))


# =============================================================================
# KNN - mais lento que SGDC porém como melhores resultados - Binário - Oito/não Oito
# =============================================================================
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train_oito)


y_val_pred = knn_clf.predict(X_val)
y_val_probas = knn_clf.predict_proba(X_val)
y_val_scores = y_val_probas[:,1]

print(classification_metrics(knn_clf, y_val_oito, y_val_pred, y_val_scores))

# gerando dados para análise das metricas
precisions, recalls, thresholds = precision_recall_curve(y_val_oito, y_val_scores)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.show()

# outra medida é a curva ROC
fpr, tpr, trhesholds = roc_curve(y_val_oito, y_val_scores)

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)
plt.show()

# =============================================================================
# Multiclass - SVC -  0 a 9
# =============================================================================
from sklearn.svm import SVC

# rodar apenas uma vez, leva muito tempo para finalizar
svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_pixel_train, y_train) 

keep_model(svm_clf, 'svm_clf_MNIST')

svm_clf = load_model('svm_clf_MNIST')

y_val_pred = svm_clf.predict(X_pixel_val)
y_val_scores = svm_clf.decision_function(X_pixel_val)

print(classification_metrics(svm_clf, y_val, y_val_pred, y_val_scores))

