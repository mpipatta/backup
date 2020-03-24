import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json, load_model
import os
import h5py
np.random.seed(7)

#temperature data after row#6800 are off
n=6800

url ='https://raw.githubusercontent.com/mpipatta/mpipatta.github.io/master/testdata/Cham5_Load.csv'
df = pd.read_csv(url, index_col=[0], parse_dates=[0])
df['Lt-1'] = df['Load'].shift(1)
df['W']=df.index.dayofweek
df['H']=df.index.hour

scaler = MinMaxScaler(feature_range=(0,1))
data=scaler.fit_transform(df[:n])


trainY = data[1:n,0]  #(starting from row1 to remove nan value) 
trainX = data[1:n,1:]

from keras.optimizers import Adam
opt = Adam(lr=0.005)

model = Sequential()
model.add(Dense(6, activation='sigmoid', input_dim=4))#
model.add(Dense(6, activation='sigmoid'))#
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(trainX, trainY, validation_split=0.1, epochs=300, batch_size=24, verbose=1)

# Save your model
#serialize model to Json
#model_json = model.to_json()
#with open('modelANN.json', 'w') as json_file:
#    json_file.write(model_json)

#serialize weights to HDF5
#model.save_weights("modelANN.h5")
#print("Saved model to disk")

model.save('modelANN.h5')

label = ['Lmin','Lmax','Tmin','Tmax','Wmin','Wmax','Hmin','Hmax']
minmax = [df['Load'].min(), df['Load'].max(), df.iloc[:6800,1].min(), df.iloc[:6800,1].max(), df['W'].min(), df['W'].max(), df['H'].min(), df['H'].max()]
df1 = pd.DataFrame(minmax, label)
df1.to_csv('minmax.csv', header=None)



