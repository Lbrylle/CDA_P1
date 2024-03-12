import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

def prepData(path):
    data = pd.read_csv('case1Data.txt', sep=',', skipinitialspace=True)
    
    data_cat = data[['C_ 1', 'C_ 2', 'C_ 3', 'C_ 4', 'C_ 5']]
    
    scaler_filename = 'scaler.pkl'
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    
    joblib.dump(scaler, scaler_filename)
        
    data_encoded = pd.get_dummies(data_cat).astype(int)

    data_norm = data.copy()
    data_norm.iloc[:, :96] = scaler.fit_transform(data_norm.iloc[:, :96])
    
    #data_tot = pd.concat([data_norm.iloc[:,:96],data_encoded], axis=1)
    data_tot = data_norm
    return data_tot