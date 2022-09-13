# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 14:31:37 2022

@author: LXY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def Data():
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[df['Name'] == 'AAPL']
    return df


def ShowDf(df):
    df.plot(x='date', y='close', kind='line', figsize=(20, 6), rot=20)


def List(df):
    FullData = df[['close']].values
    sc = MinMaxScaler()
    DataScaler = sc.fit(FullData)
    X = DataScaler.transform(FullData)
# split into list
    X_samples = list()
    y_samples = list()
    NumerOfRows = len(X)
    TimeSteps = 20

# Iterate thru the values to create combinations
    for i in range(TimeSteps, NumerOfRows, 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i]
        X_samples.append(x_sample)
        y_samples.append(y_sample)

# Reshape the Input as a 3D (number of samples, Time Steps, Features)
    X_data = np.array(X_samples)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)

# We do not reshape y as a 3D data as it is supposed to be a single column only
    y_data = np.array(y_samples)
    y_data = y_data.reshape(y_data.shape[0], 1)

# Splitting the data into training and testing
# Choosing the number of testing data records
    TestingRecords = 250
    X_train = X_data[:-TestingRecords]
    X_test = X_data[-TestingRecords:]
    y_train = y_data[:-TestingRecords]
    y_test = y_data[-TestingRecords:]

    num = []
    num.append(X_train)
    num.append(X_test)
    num.append(y_train)
    num.append(y_test)
    return num


def Model(df):
    num = List(df)
    X_train = num[0]
    y_train = num[2]
# Defining Input shapes for LSTM
    TimeSteps = X_train.shape[1]
    TotalFeatures = X_train.shape[2]

# Initialising the RNN
    regressor = Sequential()

# Adding the First input hidden layer and the LSTM layer
# return_sequences=True,
# means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units=10, activation='relu',
                       input_shape=(TimeSteps, TotalFeatures),
                       return_sequences=True))

# Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units=5, activation='relu',
                       input_shape=(TimeSteps, TotalFeatures),
                       return_sequences=True))

# Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units=5, activation='relu', return_sequences=False))

# Adding the output layer
    regressor.add(Dense(units=1))

# Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')

# Measuring the time taken by the model to train
    StartTime = time.time()

# Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, batch_size=5, epochs=100)
    EndTime = time.time()
    print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')
    return regressor


def Accuracy_Test(df):
    num = List(df)
    X_train = num[0]
    X_test = num[1]
    y_test = num[3]
    TestingRecords = 250
    TimeSteps = 20

    FullData = df[['close']].values
    sc = MinMaxScaler()
    DataScaler = sc.fit(FullData)

    regressor = Model(df)
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)

# Getting the original price values for testing data
    orig = y_test
    orig = DataScaler.inverse_transform(y_test)

# Accuracy of the predictions
    print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())
    plt.figure()
    plt.plot(predicted_Price, color='blue', label='Predicted Volume')
    plt.plot(orig, color='lightblue', label='Original Volume')
    plt.title('Stock Price Predictions')
    plt.xlabel('Date')
    plt.xticks(range(TestingRecords), df.tail(TestingRecords)['date'])
    plt.ylabel('Apple Stock Price')
    plt.grid(True)
    fig = plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(6)
    ax = fig.add_subplot(111)
    tick_spacing = 15
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.legend()
    plt.show()

    TrainPredictions = DataScaler.inverse_transform(regressor.predict(X_train))
    TestPredictions = DataScaler.inverse_transform(regressor.predict(X_test))

    FullDataPredictions = np.append(TrainPredictions, TestPredictions)
    FullDataOrig = FullData[TimeSteps:]
    re = []
    re.append(FullDataPredictions)
    re.append(FullDataOrig)
    return re


def Accuracy(df):
    re = Accuracy_Test(df)
    FullDataPredictions = re[0]
    FullDataOrig = re[1]
# plotting the full data
    plt.figure()
    plt.plot(FullDataPredictions, color='blue', label='Predicted Price')
    plt.plot(FullDataOrig, color='lightblue', label='Original Price')
    plt.title('Stock Price Predictions')
    plt.ylabel('Stock Price')
    plt.legend()
    fig = plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(8)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    df = Data()
#   ShowDf(df)
#   List(df)
#    Accuracy_Test(df)
    Accuracy(df)










