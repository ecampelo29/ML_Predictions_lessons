#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:00:57 2021

@author: ecampelo
"""

# importações necessárias
import os 
import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import DBSCAN

#from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

#import joblib
import urllib

# settings
# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'Resultados_DBSCAN.txt')
fig_path = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'ML_DBScan')

# capturando a imagem usada no livro para mapear os imóveis na califórnia
images_path = relatorio = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning')
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))

# =============================================================================
# Funções
# =============================================================================
# def plot_Anomalies_3D (scan, X, axes, title):
#     ax = fig = []
#     core_mask = np.zeros_like (scan.labels_, dtype=bool) # cria uma array de false 
#     core_mask[scan.core_sample_indices_] = True # deixa como falso apenas os outliers
#     anomalies_mask = scan.labels_ == -1 # marca os casos de anomalia
#     non_core_mask = ~(core_mask | anomalies_mask) # marca os casos não core
    
#     cores = scan.components_
#     anomalies = X[anomalies_mask]
#     non_cores = X[non_core_mask]
    
#     # são 6 pontos e estou imprimindo apenas 3 no Axes3d
    
#     axes_lim = [int(np.floor(min(cores[:, axes[0]]))), int(np.ceil(max(cores[:, axes[0]]))),
#                 int(np.floor(min(cores[:, axes[1]]))), int(np.ceil(max(cores[:, axes[1]]))),
#                 int(np.floor(min(cores[:, axes[2]]))), int(np.ceil(max(cores[:, axes[2]])))]
    
#     fig = plt.figure(figsize = (7,4))
#     # ax = fig.add_subplot(111, projection = '3d')
#     ax = Axes3D(fig)
#     ax.view_init(10, -60)
#     plt.scatter(cores[:, axes[0]], cores[:, axes[1]], cores[:,axes[2]],
#                 c=scan.labels_[core_mask], marker='o', cmap="Paired")
    
    
#     plt.scatter(cores[:, axes[0]], cores[:, axes[1]], cores[:,axes[2]], marker='*',  
#                     c=scan.labels_[core_mask])
#     plt.scatter(anomalies[:, axes[0]], anomalies[:,axes[1]], anomalies[:,axes[2]],
#                     c="r", marker="x")
#     plt.scatter(non_cores[:, axes[0]], non_cores[:, axes[1]],
#                 non_cores[:, axes[2]],  c='orange', marker=".")
#     plt.title("eps={:.2f}, min_samples={} \n {} x {} x {}".format(
#                 scan.eps, scan.min_samples, title[0],title[1], title[2]), 
#               fontsize=14)
#     ax.set_xlim(axes_lim[0:2])
#     ax.set_ylim(axes_lim[2:4])
#     ax.set_zlim(axes_lim[4:6])
    
#     plt.show()


def plot_Anomalies_2D (scan, X, axes, title):
    # são 6 pontos e estou imprimindo apenas 2 no scatter
    core_mask = np.zeros_like (scan.labels_, dtype=bool) # cria uma array de false 
    core_mask[scan.core_sample_indices_] = True # deixa como falso apenas os outliers
    anomalies_mask = scan.labels_ == -1 # marca os casos de anomalia
    non_core_mask = ~(core_mask | anomalies_mask) # marca os casos não core
    
    cores = scan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    axes_lim = [int(np.floor(min(cores[:, axes[0]]))), int(np.ceil(max(cores[:, axes[0]]))),
                int(np.floor(min(cores[:, axes[1]]))), int(np.ceil(max(cores[:, axes[1]])))]
    
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    
    plt.scatter(cores[:, axes[0]], cores[:, axes[1]],
                c=scan.labels_[core_mask], marker='o', cmap="Paired")
    
    
    plt.scatter(cores[:, axes[0]], cores[:, axes[1]], marker='*',  
                    c=scan.labels_[core_mask])
    plt.scatter(anomalies[:, axes[0]], anomalies[:,axes[1]],
                    c="r", marker="x")
    plt.scatter(non_cores[:, axes[0]], non_cores[:, axes[1]],
                c='orange', marker=".")
        
    ax.set_xlim(axes_lim[0:2])
    ax.set_ylim(axes_lim[2:4])
    
    
    plt.title("eps={:.2f}, min_samples={} \n {} x {}".format(
                scan.eps, scan.min_samples, title[0],title[1]), 
              fontsize=14)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()

# marcando as labels no data set para geração do mapa
# ideia tirada deste paper: https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d                  
def plot_map(house_clusters, house_anomalies, house_scan, var_analysis):
    # número de clusters encontrado
    n_clusters_ = len(set(house_scan.labels_)) - (1 if -1 in house_scan.labels_
                                                      else 0)
    # mostrando os dados no mapa da california
    california_img=mpimg.imread(os.path.join(images_path, filename))
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    plt.scatter(x=house_clusters["longitude"], y=house_clusters["latitude"], 
                          c=house_clusters["labels"], cmap=plt.get_cmap("jet"),
                           alpha=0.7
                          )    
    plt.scatter(x=house_anomalies["longitude"], y=house_anomalies["latitude"], 
                          c='red', cmap=plt.get_cmap("jet"), marker='x')    
    
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
               cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    plt.title("eps={:.2f}, min_samples={} \n clusters {} analisys: {}"
              "\n anomalies: {}".format(
                  house_scan.eps, house_scan.min_samples, n_clusters_,
                  var_analysis, len(house_anomalies)), fontsize=14)
    plt.savefig(fig_path+'/ML_DBScan_'+var_analysis, format='png', dpi=300)
    plt.show()
# =============================================================================
# Objetos
# =============================================================================
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
# =============================================================================
# Processando os dados
# =============================================================================        
download_files = os.path.join(os.path.expanduser('~'), 'Documentos', 
                                  'Machine learning', 'Data_files')
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
# median_house_value -> tem um acúmulo/teto em 500mil
# =============================================================================
# Como a idéia é analisar outlier no preço médio, vou tirar os valores que 
# estão no cap de 500mil que existe no arquivo
# Também serão retirados os median_age que tem cap de 52
# =============================================================================
housing = housing[(housing['median_house_value'] < 500000) & 
                  (housing['housing_median_age'] < 52)]

# correlações
corr = housing.corr()
corr['median_house_value'].sort_values(ascending=False)

house_train = housing.copy()
# unsupervised não tem teste, por que não tem o que se comparar
# separando dados para teste e treino de forma aleatória com proporção de 80/20
# house_train, house_test = train_test_split(housing, test_size = 0.2, random_state=42)
#['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       # 'total_bedrooms', 'population', 'households', 'median_income',
       # 'median_house_value', 'ocean_proximity']

# variáveis a serem comparadas em relação às espaciais
list_var_analysis = ['housing_median_age', 'total_rooms', 'total_bedrooms', 
                     'population', 'households', 'median_income',
                     'median_house_value']

for var in list_var_analysis: 
    num_attrib = ['longitude', 'latitude'] + [var]    
    cat_attrib = [ 'ocean_proximity']
    full_attrib = num_attrib+cat_attrib
    
    num_pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ])    
    
    cat_pipe = Pipeline ([('binner', qbinner(cat_attrib)),])
    
    full_pipeline = ColumnTransformer([
            ("num", num_pipe, num_attrib),
            ("cat", cat_pipe, cat_attrib),
        ])
    
    
    house_train_prep = full_pipeline.fit_transform(house_train[full_attrib])

# house_train_prep = num_pipe.fit_transform(house_train[num_attrib])

# =============================================================================
# variando eps de 0.5 a 1 há melhora na identificação de possíveis anomalia, 
# porém alguns pontos no scatter que poderiam indicar anomalias deixam de ser 
# marcados, visto que aumentamos a distância para que esse ponto possa pertencer
# ou não a um núcleo. Desta forma, temos pontos não núcleos por não terem
# a quantidade mínima para se juntar ao cluster, mais ainda sim não podem
# ser considerados como anomalia. 
# Reduzindo o min_sample diminiu a marcação, chegando a praticamente 0
# quando min_sample tende a 1, porque essa é quantidade de vizinhos que estão
# dentro da distância eps, logo menos vizinhos, menos núcleos construídos.
# =============================================================================

# o eps e min foi setado sobre as variáveis posicionais latitude, longitude e 
# ocean proximity, de forma a identificar os clusters regionais. 
# após isto inlcuí uma variável extra por vez para analisar suas anomalias de 
# forma independente

    house_scan = DBSCAN(eps=0.5, min_samples=9) 
    
    house_scan.fit(house_train_prep)
    
    house_scan.labels_[:10]
    
    # marca os clusters
    house_train = house_train.copy()
    house_train['labels'] = pd.Series(house_scan.labels_).values
    house_clusters =  house_train[house_train['labels'] != -1]
    # isola as possíveis anomalias
    house_anomalies = house_train[house_train['labels'] == -1]
    ############
    # scan = house_scan
    # X = house_train_prep
    #columns_trained = full_attrib
    # axes3d = [4,5,2]
    #axes2d= [0, 1]

    plot_map(house_clusters, house_anomalies, house_scan, var)

                  
# análise de 3 pontos
# plot_Anomalies_3D(house_scan, house_train_prep, axes3d,
#                   [columns_trained[axes3d[0]], 
#                    columns_trained[axes3d[1]],
#                    columns_trained[axes3d[2]]])

# análise de 2 pontos
# plot_Anomalies_2D(house_scan, house_train_prep, axes2d, 
#                   [columns_trained[axes2d[0]], 
#                    columns_trained[axes2d[1]]])
# conclusões
# =============================================================================
# Training 1 Longitude x Latitude: O gráfico mostra que existem 3 grupos 
# (eps=0.2 min_samples=15) com indicação de vários outliers que estão a margem 
# do conglomerado principal vide imagem ML_DBScan Training1. Os outliers 
# baseados exclusivamente em suas coordenadas são plausíveis e deveriam ser 
# analisados, se esse fosse o intuito.
# Training 2 Long x Lat x Ocean_Proximity: após ajustes em eps e min_samples 
# (eps=30 e min_sample = 10), temos um resultado satisfatório para a análise de 
# outliers posicionais, ou seja, os agrupamentos baseados em coordenadas e 
# localização mostram apenas alguns individuos a serem analisados como
# possíveis problemas, se este fosse o intuito. 
# T3 Posicionais + median_house_value: Se considerarmos que o que temos no
# Training 2 (T2) é razoável para uma análise posicional, quando acrescentamos
# o valor médio dos imóveis, há um aumento significativo de outliers. Neste caso
# há indicação de que imóveis próximos à São Franciso, Sacramento e ao sudeste 
# de Los Angeles devam ser estudados, pois seus valores médios não estariam 
# dentro do esperado. 
# T4 Median_house_value x Median_incame: analisando as possíveis anomalias entre
# estas duas variáveis, verifica-se que estão concentradas nos grandes centros 
# Losangeles e São Francisco.
# T5 Median_house_value x Total_rooms: novamente os dois maiores centros populacionais
# são destaque nas anomalias. 
# T6 Median_house_values x rooms: como as variáveis sobre quartos apresentavam
# o mesmo comportamento, foi possível incorporá-las no mesmo modelo. 
# =============================================================================

# =============================================================================
# Usando a abordagem de se analisar uma varíavel numérica em relação às espaciais
# (lat, long, ocean proximity) verifica-se que:
# - median_house_value -> 10 anomalias, logo a precificação por localização é aceitável
# - housing_median_age -> 22 anomalias, também aceitável 
# - median_income -> 51 anomalias, concentradas nos grandes centros. Necessário estudá-las para entender a motivação
# Outras que estão ligadas às concentrações dos grandes centros urbanos, que seria interessante estudá-las:
# - population -> 145 anomalias
# - households -> 132 anomalias
# - total_bedrooms -> 148 anomalias
# - total_rooms -> 177 anomalias    
# Quando isoladas, essas variáveis apresentam fortes correlações positivas entre si. 
# housing[['total_rooms', 'total_bedrooms', 'population', 'households']].corr()
#                total_rooms  total_bedrooms  population  households
# total_rooms        1.000000        0.935415    0.860279    0.922493
# total_bedrooms     0.935415        1.000000    0.880314    0.978815
# population         0.860279        0.880314    1.000000    0.910656
# households         0.922493        0.978815    0.910656    1.000000
# =============================================================================

to_do: experimentar com redução de dimensionalidade (PCA e SVD), já que 
dbscan tem limitações por conta da medição espacial. 



