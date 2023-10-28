#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from stock_market_strategy_analysis.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

#%%  Function - Drop columns, Update column names & Update frequency of index
def data_preparation(df):

    df = df.drop(['series ', '52W H ', '52W L ', 'VALUE ', 'No of trades '],
                       axis=1)
    df = df.rename(columns={'OPEN ': 'open', 'HIGH ': 'high', 'LOW ': 'low', 'PREV. CLOSE ': 'prev_close',
                                  'ltp ': 'ltp', 'close ': 'close', 'vwap ': 'vwap', 'VOLUME ': 'volume'})

    df = df.drop_duplicates()
    df = df.resample('D').asfreq()
    df = df.interpolate('linear')

    return df

#%%Read the data and prepare

data = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/axisbank.csv",
                   index_col=0, parse_dates=[0], dayfirst=True)

axis_df = data_preparation(data)

#%% add technical indicators

# RSI - Relative Strength Index
axis_df['rsi'] = calculate_rsi(axis_df['close'], 14)

# MACD - Moving Average Convergence Divergence
macd_signal = calculate_macd(axis_df['close'], 12, 26, 9)
axis_df['macd_line'] = macd_signal[0]
axis_df['signal_line'] = macd_signal[1]

# Bollinger bands
bands = calculate_bollinger_bands(axis_df['close'], 30, 2)
axis_df['upper_bollinger_band'] = bands[0]
axis_df['lower_bollinger_band'] = bands[1]


