# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:33:02 2022

@author: LXY

"""

import pandas as pd
import pandas_ta as ta
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def Data():
    df = pd.read_csv('all_stocks_5yr.csv')
    df = df[df['Name'] == 'AAPL']
    df.set_index(pd.DatetimeIndex(df['date']), inplace=True)
    df = df[['close']]
    df.ta.ema(close='close', length=10, append=True)
    df = df.iloc[10:]
    return df


def Draw_EMA(df):
    x1 = pd.read_csv('all_stocks_5yr.csv')
    x1 = x1[x1['Name'] == 'AAPL']
    use = x1['date']
    use = use.iloc[10:]
    plt.figure(figsize=(18, 8))
    plt.plot(use, df['close'], label="close", linestyle="-.")
    plt.plot(use, df['EMA_10'], label="EMA_10", linestyle="--")
    plt.legend()
    plt.title("The plot of $TSLA historic pricing from 2013-2018 with the EMA overlaid.", fontsize=20)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Number", fontsize=14)
    plt.xticks(use, rotation=90)
    x = MultipleLocator(8)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x)
    plt.show(block=True)


def LinearModle(df):
    X_train, X_test, y_train, y_test = train_test_split(df[['close']], df[['EMA_10']], test_size=.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
# Results
    coef = model.coef_
    mae = mean_absolute_error(y_test, y_pred)
    coefficient_of_determination = r2_score(y_test, y_pred)
    return model


def ErrorShow(df):
    X_train, X_test, y_train, y_test = train_test_split(df[['close']], df[['EMA_10']], test_size=.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, label="prediction")
    plt.scatter(X_test, y_test, label="real number", color='red')
    plt.legend()
    plt.title("The contradistinction", fontsize=20)
    plt.show(block=True)
 

if __name__ == '__main__':
    df = Data()
    #Draw_EMA(df)
    #LinearModle(df)
    ErrorShow(df)
