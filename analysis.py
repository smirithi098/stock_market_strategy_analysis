#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_rsi, calculate_ema

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

#%% function to add technical indicators to df

def stochastic_rsi_with_ema(df, close_price):

    # # RSI - Relative Strength Index
    # df['rsi'] = calculate_rsi(close_price, 14)
    #
    # # MACD - Moving Average Convergence Divergence
    # macd_signal = calculate_macd(close_price, 12, 26, 9)
    # df['macd_line'] = macd_signal[0]
    # df['signal_line'] = macd_signal[1]
    #
    # # Bollinger bands
    # bands = calculate_bollinger_bands(close_price, 30, 2)
    # df['upper_bollinger_band'] = bands[0]
    # df['lower_bollinger_band'] = bands[1]

    # stochastic_rsi
    df['stoch_rsi'] = calculate_stochastic_rsi(close_price, 13)

    # EMA - Exponential Moving Average
    df['ema_50'] = calculate_ema(close_price, 50)
    df['ema_100'] = calculate_ema(close_price, 100)

#%% call the function to get the technical indicators

stochastic_rsi_with_ema(axis_df, axis_df['close'])

#%% Filter out rows with value not null

start_index = axis_df.index.get_loc('2006-05-01')
end_index = axis_df.index.get_loc('2023-05-31')
all_data = axis_df.loc[axis_df.index[start_index]:axis_df.index[end_index], :]
"""
So go to the indicator settings and change the length from 9 to 50. 
And here, the stochastic RSI crossed the 80 level to the downside, indicating a sell signal, 
and the price is also closed below the 50 EMA. Both EMAs are sloped downwards, 
and the price is also closed below the 50 EMA. Here, the stochastic RSI crossed the 20 level to the upside,
 indicating a buy signal, and the 50 EMA is above the 100 EMA, and the price is also closed above the 50 EMA.
"""

#%% calculate the buy-sell points in the data

all_data[:]['signal_1'] = np.where((all_data['stoch_rsi'] > 20) & (all_data['stoch_rsi'] <= 50) & (all_data['ema_50'] > all_data['ema_100']), 1, 0)
all_data[:]['buy_sell_position'] = all_data['signal_1'].diff()

#%% visualize the buy sell points with the technical indicators in place

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 6), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(all_data.close, color='black', label='Close price', linewidth=1)
ax1.plot(all_data.ema_50, color='orange', label='50-day EMA', linewidth=1)
ax1.plot(all_data.ema_100, color='blue', label='100-day EMA', linewidth=1)
ax1.set_xlim([all_data.index[0], all_data.index[-1]])

ax2.plot(all_data.stoch_rsi, color='red', linewidth=0.8)
ax2.axhline(y=20, color='darkgrey', linestyle='-')
ax2.axhline(y=80, color='darkgrey', linestyle='-')
ax2.set_ylim([0, 100])

plt.title("Stochastic RSI with 50 & 100 day EMA strategy")
plt.legend()
plt.tight_layout()
plt.show()
