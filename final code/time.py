# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:16:03 2022

@author: LXY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from pmdarima.arima import auto_arima


def Data():
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[df['Name'] == 'AAPL']
    return df


def Split(df):
    test_data = df[df['date'] >= '2017-06-01']

    # plotting the data
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    plt.grid(True)
    plt.xlabel('Dates')
    plt.ylabel('Closing Prices')
    plt.plot(df['date'], df['close'], 'green', label='Train data')
    plt.plot(test_data['date'], test_data['close'], 'blue', label='Test data')
    tick_spacing = 200
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.legend()
    plt.show()


def InterpretModel(df):
    train_data = df[df['date'] < '2017-06-01']
    model_autoARIMA = auto_arima(y=train_data['close'],
                                 start_p=0,
                                 start_q=0,
                                 test='adf',
                                 max_p=3,
                                 max_q=3,
                                 m=1,
                                 d=None,
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=True)
# The summary function helps us summarize all the findings from the model.
#    print(model_autoARIMA.summary())
    model_autoARIMA.plot_diagnostics(figsize=(15, 8))
    plt.show()


def TrainModel(df):
    train_data = df[df['date'] < '2017-06-01']
    test_data = df[df['date'] >= '2017-06-01']
    model = ARIMA(train_data['close'], order=(1, 1, 0))
# using p=1, d=1, q=0
    fitted = model.fit(disp=-1)
# display < 0 means no output to be printed
    fc, se, conf = fitted.forecast(test_data.shape[0], alpha=0.05)

    fc_series = pd.Series(fc, index=test_data['date'])
    lower_series = pd.Series(conf[:, 0], index=test_data['date'])
    upper_series = pd.Series(conf[:, 1], index=test_data['date'])

    # plot all the series together
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    plt.plot(train_data['date'], train_data['close'], label='Training data')
    plt.plot(test_data['date'], test_data['close'],
             color='blue', label='Actual Stock Price')
    plt.plot(fc_series, color='orange', label='Predicted Stock Price')
    plt.fill_between(lower_series.index, lower_series, upper_series,
                     color='k', alpha=.10)
    tick_spacing = 200
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Apple Stock Price')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    return fc


def Error(df):
    fc = TrainModel(df)
    test_data = df[df['date'] >= '2017-06-01']
    print("\n")
    print("\n")
    mse = mean_squared_error(test_data['close'], fc)
    print('MSE: '+str(mse))
    mae = mean_absolute_error(test_data['close'], fc)
    print('MAE: '+str(mae))
    rmse = math.sqrt(mean_squared_error(test_data['close'], fc))
    print('RMSE: '+str(rmse))
    mape = np.mean(np.abs(fc - test_data['close'])/np.abs(test_data['close']))
    print('MAPE: '+str(mape))


if __name__ == '__main__':
    df = Data()
    Split(df)
    InterpretModel(df)
    Error(df)
