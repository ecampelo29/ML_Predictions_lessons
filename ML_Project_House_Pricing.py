# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

# importações necessárias
import os 
import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

import joblib


# settings
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'Resultados_ML.txt')

# classes
# criando um clipper para uso no pipeline (deve ter fit e transform)
class clipper(BaseEstimator, TransformerMixin):
    def __init__ (self, lista_clipping):
        self.lista_clipping = lista_clipping
        
    def fit (self, X, y=None):
        return self # faz nada

    def transform (self, X):
        for feature, cap in self.lista_clipping:
            X.loc[X[feature] > cap, feature] = cap
        return np.array(X)

# criando um binner para uso no pipeline (deve ter fit e transform)
class qbinner(BaseEstimator, TransformerMixin):
    def __init__ (self, lista_binner):
        self.lista_binner = lista_binner # variáveis a serem transformadas
        self.cat_encoder = OneHotEncoder()
        self.new_x = None
        
    def fit (self, X, y=None):
        return self # faz nada

    def transform (self, X):
        for cat in self.lista_binner:
            try: 
            # categorical quartile bucketing 
                X[cat] = pd.qcut(X[cat],  4 , labels=['q1','q2','q3','q4'])
            except TypeError:
                continue
            
        new_x = X[self.lista_binner]
        
        return self.cat_encoder.fit_transform(new_x)
    
# funções
# retirada do mateiral do livro para análise da curva de aprendizagem
def plot_learning_curves(model, X_train, X_val, y_train, y_val):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), 500):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown

# grava os resultados das análises em texto para comparações 
# RMSE - quanto de erro está sendo gerado pelo modelo (zero: sem erro)
def linear_metrics (model, X_train, X_test, y_train, y_test, features, relatorio=relatorio):
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    resultado= ['Treinamento do modelo {} com as variáveis: '.format(model), 
                ','.join(features), 
                'MSE_train: ' + str(mse_train) + '  MSE_test: '+str(mse_test), 
                'R²_train: ' + str(r2_train) + '  R²_test: '+str(r2_test),
                'RMSE_train: ' + str(np.sqrt(mse_train))+'  RMSE_test: ' + str(np.sqrt(mse_test))]
    
    if os.path.isfile(relatorio):
        with open (relatorio, 'a') as r:
            r.write('\n\n'+'\n'.join(resultado))
    else: 
        with open (relatorio, 'w') as r:
            r.write('\n\n'+'\n'.join(resultado))
            
    return ('\n\n'+'\n'.join(resultado), y_pred_train, y_pred_test)

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
    

# variáveis           
download_files = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'Data_files')
# =============================================================================
# Processando os dados
# =============================================================================
# lendo arquivo
housing = pd.read_csv(download_files+'/housing.csv')

# explorando arquivo
housing.info()

# existem 207 registros sem dados de total_rooms
housing.dropna(inplace=True)

# analise geral
desc = housing.describe()
# total_bedrooms/total_rooms/population aparenta ter outliers extremos
# descobrir a razão do median_income (muito baixo para salário médio)

# distribuições
housing.hist(bins=50, figsize=(30,15))
# housing_median_age -> tem um acúmulo/teto em entorno de 55
# median_house_value -> tem um acúmulo/teto em 50mil

# correlações
corr = housing.corr()
corr['median_house_value'].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

# criando novas features a partir de existentes (baseando no livro)
ratios = [('bedoom_per_room', 'total_bedrooms', 'total_rooms'), # quartos por comodos
          ('household_per_room', 'total_rooms', 'households'), # comodos por família
          ('household_per_population', 'population', 'households') # densidade de famílias
          ]

for new_col, numerator, denominator in ratios:
    housing[new_col] = housing[numerator]/housing[denominator]

# separando dados para teste e treino de forma aleatória com proporção de 80/20
house_train, house_test = train_test_split(housing, test_size = 0.2, random_state=42)


# transformando features
features = ['total_rooms', 'total_bedrooms', 'median_income', 'population', 
            'households', 'housing_median_age', 'ocean_proximity', 
            'household_per_room', 'household_per_population', 'bedoom_per_room']
num_atrib = ['total_rooms', 'total_bedrooms', 'median_income', 
             'housing_median_age', 'population', 'households', 
              'household_per_room', 'household_per_population', 'bedoom_per_room']
cat_atrib = [ 'ocean_proximity']
label = ['median_house_value']

house_train_transformed = house_train[features].copy()
house_test_transformed = house_test[features].copy()
train_label = house_train[label].copy()
test_label = house_test[label].copy()

# aplicando cliping por conta de outliers importantes 
cliping = [('total_rooms',10000) , ('total_bedrooms', 2000), 
           ('population', 5000), ('household_per_room', 15), 
           ('household_per_population', 10)
           ]

# chama o pipe para transformações dos dados
num_pipe = Pipeline([
                ('clipper', clipper(cliping)),
                ('scaler', StandardScaler()),
                ])    


cat_pipe = Pipeline ([('binner', qbinner(cat_atrib)),])



full_pipeline = ColumnTransformer([
        ("num", num_pipe, num_atrib),
        ("cat", cat_pipe, cat_atrib),
    ])

# gera um array com dados preparados para modelagem
house_train_prep = full_pipeline.fit_transform(house_train_transformed)
# não usar fit_transform no test set
house_test_prep = full_pipeline.transform(house_test_transformed)

# =============================================================================
# Treinando regressão linear - Resultados não são bons
# =============================================================================
   
# escolhendo o modelo
# regressão linear para predições

lin_regr = LinearRegression()
lin_regr.fit(house_train_prep, train_label)



# guardando as métricas do modelo atual
metricas, y_pred_train, y_pred_test = linear_metrics(lin_regr, house_train_prep, house_test_prep, 
                               train_label, test_label, features)
    
print(metricas)
# plot_learning_curves(lin_regr, house_train_transformed, house_test_transformed, train_label, test_label)

# analisando o treinamento por setores dos dados
# ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 
# 'f1_micro', 'f1_samples', 'f1_weighted', 'log_loss', 'mean_absolute_error', 
# 'mean_squared_error', 'median_absolute_error', 'precision',   
# 'precision_macro', 'precision_micro', 'precision_samples', 
# 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 
# 'recall_samples', 'recall_weighted', 'roc_auc']

lin_scores = cross_val_score(lin_regr, house_train_prep, train_label,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

lin_scores = cross_val_score(lin_regr, house_train_prep, train_label,
                             scoring="r2", cv=10)
display_scores(lin_scores)


# =============================================================================
# Usando forest conforme livro - Melhores resultados
# =============================================================================
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(house_train_prep, np.ravel(train_label))

housing_predictions = forest_reg.predict(house_train_prep)
forest_mse = mean_squared_error(train_label, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

forest_scores = cross_val_score(forest_reg, house_train_prep, train_label,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# =============================================================================
# USando SVM - conforme livro - Resultado não são bons
# =============================================================================
# variação do Support Vector, porém para uso em poucos dados, pelo custo computacional
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(house_train_prep,  np.ravel(train_label))
housing_predictions = svm_reg.predict(house_train_prep)
svm_mse = mean_squared_error(train_label, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse


metricas, y_pred_train, y_pred_test = linear_metrics(svm_reg, house_train_prep, house_test_prep, 
                               train_label, test_label, features)
    
print(metricas)
# =============================================================================
# Usando parametrização do livro para ver comportamento do GridSearch
# =============================================================================
# deixando o sistema procurar por um modelo através de vários parâmetros 
# pré estabelecidos, que são combinados entre si
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
housing_labels = np.ravel(train_label)

grid_search.fit(house_train_prep, housing_labels)

grid_search.best_params_

best_rdforest = grid_search.best_estimator_

housing_predictions = best_rdforest.predict(house_train_prep)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

metricas, y_pred_train, y_pred_test = linear_metrics(best_rdforest, house_train_prep, house_test_prep, 
                               train_label, test_label, features)
    
print(metricas)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)



# =============================================================================
# Guardando e carregando o modelo
# =============================================================================
# guardando o modelo
# keep_model(lin_regr, 'regressao_final')
# model_loaded = load_model("regressao_final")  


# analizando alguns resultados 
alguns_casos = house_train_prep[:5]
algumas_labels = train_label[:5]

print('predictions', lin_regr.predict(alguns_casos)) # bem longe em alguns casos
print('predictions', model_loaded.predict(alguns_casos)) # bem longe em alguns casos


