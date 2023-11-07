#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_rsi

#%%  Function - Drop columns, Update column names & Update frequency of index
def data_preparation(df):

    df = df.drop(['series ', '52W H ', '52W L ', 'VALUE ', 'No of trades '],
                       axis=1)
    df = df.rename(columns={'OPEN ': 'open', 'HIGH ': 'high', 'LOW ': 'low', 'PREV. CLOSE ': 'prev_close',
                                  'ltp ': 'ltp', 'close ': 'close', 'vwap ': 'vwap', 'VOLUME ': 'volume'})

    df = df.drop_duplicates()
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').asfreq()
    df = df.interpolate('linear')

    return df

#%%Read the data and prepare

data = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/axisbank.csv",
                   index_col=0, parse_dates=[0], dayfirst=True)

axis_df = data_preparation(data)

#%% Read the strategy data

strategies = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/strategies.csv")

#%% function to add technical indicators to df

def add_indicators(df, close_price):

    # RSI - Relative Strength Index
    df['rsi'] = calculate_rsi(close_price, 14)

    # MACD - Moving Average Convergence Divergence
    macd_signal = calculate_macd(close_price, 12, 26, 9)
    df['macd_line'] = macd_signal[0]
    df['signal_line'] = macd_signal[1]

    # Bollinger bands
    bands = calculate_bollinger_bands(close_price, 30, 2)
    df['upper_bollinger_band'] = bands[0]
    df['lower_bollinger_band'] = bands[1]

    # stochastic_rsi
    df['stoch_rsi'] = calculate_stochastic_rsi(close_price, 13)

#%% call the function to get the technical indicators

add_indicators(axis_df, axis_df['close'])

