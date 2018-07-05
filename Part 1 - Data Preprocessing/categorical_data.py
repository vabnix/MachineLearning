# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plot
import pandas as panda
from sklearn.preprocessing import Imputer

dataset = panda.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
#array([[0, 44.0, 72000.0],
#       [2, 27.0, 48000.0],
#       [1, 30.0, 54000.0],
#       [2, 38.0, 61000.0],
#       [1, 40.0, nan],
#       [0, 35.0, 58000.0],
#       [2, nan, 52000.0],
#       [0, 48.0, 79000.0],
#       [1, 50.0, 83000.0],
#       [0, 37.0, 67000.0]], dtype=object)

from sklearn.preprocessing import OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
