import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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

trainX = trainX.reshape((trainX.shape[0], 1, 4))

from keras.optimizers import Adam
opt = Adam(lr=0.005)

model = Sequential()
model.add(LSTM(6, activation='sigmoid',return_sequences=True, input_shape=(1, 4)))
model.add(LSTM(6, activation='sigmoid', input_shape=(1, 4)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=opt)
history = model.fit(trainX, trainY, validation_split=0.1, epochs=300, batch_size=24, verbose=1)

model.save('modelLSTM.h5')