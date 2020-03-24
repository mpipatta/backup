import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url ='https://raw.githubusercontent.com/mpipatta/mpipatta.github.io/master/testdata/Cham5_Load.csv'
df = pd.read_csv(url, index_col=[0], parse_dates=[0])
df['Lt-1'] = df['Load'].shift(1)

X=df[['Lt-1','Temp']].values
y=df['Load'].values 

regressor = LinearRegression()
#temperature data after row#6800 are off
n=6800
regressor.fit(X[1:n], y[1:n])

# Save your model
from sklearn.externals import joblib
joblib.dump(regressor, 'modelMLR.pkl')
print("Model dumped!")

# Load the model that you just saved
regressor = joblib.load('modelMLR.pkl')

