import numpy as np
import matplotlib.pyplot as plt
import pandas as panda

#Reading the file from path defining dataset
dataset = panda.read_csv('Data.csv')
#now that CSV is loaded , it can be viewed in variable explorer

X = dataset.iloc[:, :-1].values
#: means we take all the column and :-1 mean except the last column
#[['France' 44.0 72000.0]
# ['Spain' 27.0 48000.0]
# ['Germany' 30.0 54000.0]
# ['Spain' 38.0 61000.0]
# ['Germany' 40.0 nan]
# ['France' 35.0 58000.0]
# ['Spain' nan 52000.0]
# ['France' 48.0 79000.0]
#['Germany' 50.0 83000.0]
# ['France' 37.0 67000.0]]

#Lets create a dependent variable vector
Y = dataset.iloc[:,3].values
#by doing so Y contain only the value from last column
#array(['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes'],dtype=object)
