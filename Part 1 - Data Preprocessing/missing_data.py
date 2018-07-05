import numpy as np
import matplotlib.pyplot as plot
import pandas as panda
from sklearn.preprocessing import Imputer

#now lets import the csv file where we will manupulate the missing data 
dataset = panda.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values

imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

#array([['France', 44.0, 72000.0],
#       ['Spain', 27.0, 48000.0],
#       ['Germany', 30.0, 54000.0],
#       ['Spain', 38.0, 61000.0],
#       ['Germany', 40.0, 63777.77777777778],
#       ['France', 35.0, 58000.0],
#       ['Spain', 38.77777777777778, 52000.0],
#       ['France', 48.0, 79000.0],
#       ['Germany', 50.0, 83000.0],
#       ['France', 37.0, 67000.0]], dtype=object)