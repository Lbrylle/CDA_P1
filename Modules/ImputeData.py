import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

def imputeData(data, K):
    K = 6
    knn_imputer = KNNImputer(n_neighbors=K, weights='distance') 
    data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)
    
    return data_imputed