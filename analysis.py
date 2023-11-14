#%% import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_stochastic_rsi, calculate_ema, calculate_sma

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

#%% function to add technical indicators to df for strategy 1

def stochastic_rsi_with_ema(df, close_price):

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

all_data[:]['ema_diff'] = all_data['ema_50'] - all_data['ema_100']
# all_data.loc[:, 'ema_diff'] = all_data['ema_50'] - all_data['ema_100']
all_data[:]['buy_signal'] = np.where((all_data['stoch_rsi'] > 0) & (all_data['stoch_rsi'] <= 42) &
                                         (all_data['ema_50'] > all_data['ema_100']),
                                         1, 0)
# all_data.loc[:, 'buy_signal'] = np.where((all_data['stoch_rsi'] > 0) & (all_data['stoch_rsi'] <= 42) &
#                                          (all_data['ema_50'] > all_data['ema_100']),
#                                          1, 0)
all_data[:]['sell_signal'] = np.where((all_data['stoch_rsi'] >= 68) & (all_data['stoch_rsi'] < 100) &
                                          (all_data['ema_50'] < all_data['ema_100']),
                                          1, 0)
# all_data.loc[:, 'sell_signal'] = np.where((all_data['stoch_rsi'] >= 68) & (all_data['stoch_rsi'] < 100) &
#                                           (all_data['ema_50'] < all_data['ema_100']),
#                                           1, 0)

#%%

crossover_points = pd.DataFrame(all_data.loc[all_data['ema_diff'].between(-1, 1), ['ema_50', 'ema_diff']])
crossover_points['diff'] = crossover_points['ema_50'].diff()

crossover_points = crossover_points.dropna(axis=0)

#%%

for i, val in enumerate(crossover_points.loc[:, 'diff'].to_list()[:-1]):
    print(i, val)

    if val < 0:
        if (not -10 < val < 10) & (val == crossover_points.loc[crossover_points.index[i-1]:crossover_points.index[i+2], 'diff'].min())\
                & (crossover_points.loc[crossover_points.index[i], 'ema_diff'] < crossover_points.loc[crossover_points.index[i+1], 'ema_diff']):
            crossover_points.loc[crossover_points.index[i], 'position'] = 'buy'
        else:
            crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

    else:
        if (not -10 < val < 10) & (val == crossover_points.loc[crossover_points.index[i-1]:crossover_points.index[i+2], 'diff'].max())\
                & (crossover_points.loc[crossover_points.index[i-1], 'ema_diff'] > crossover_points.loc[crossover_points.index[i], 'ema_diff']):
            crossover_points.loc[crossover_points.index[i], 'position'] = 'sell'
        else:
            crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

#%%
buy_sell_data = crossover_points[(crossover_points['position'] != 'nothing')]
buy_sell_data = buy_sell_data.loc[:buy_sell_data.index[-3], :]

temp = buy_sell_data.groupby((buy_sell_data['position'] != buy_sell_data['position'].shift()).cumsum()).apply(lambda x: (x.index[0], x.index[-1]))

for tup in temp:
    if len(buy_sell_data.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_data.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_data.loc[tup[0]:tup[1], 'ema_50'].idxmin())
            buy_sell_data.loc[tup[0]:tup[1], 'position'] = buy_sell_data.loc[tup[0]:tup[1], 'position'].where(
                buy_sell_data.loc[tup[0]:tup[1], 'position'].index == min_value, 'nothing')

        elif buy_sell_data.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_data.loc[tup[0]:tup[1], 'ema_50'].idxmax())
            buy_sell_data.loc[tup[0]:tup[1], 'position'] = buy_sell_data.loc[tup[0]:tup[1], 'position'].where(
                buy_sell_data.loc[tup[0]:tup[1], 'position'].index == max_value, 'nothing')

#%% visualize the buy sell points with the technical indicators in place

buy_signals = buy_sell_data.loc[buy_sell_data['position'] == 'buy', 'ema_50']
sell_signals = buy_sell_data.loc[buy_sell_data['position'] == 'sell', 'ema_50']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(all_data.close, color='palevioletred', label='Close price', linewidth=1)
ax1.plot(all_data.ema_50, color='royalblue', label='50-day EMA', linewidth=1)
ax1.plot(all_data.ema_100, color='darkorange', label='100-day EMA', linewidth=1)
ax1.set_xlim([all_data.index[0], all_data.index[-1]])
ax1.legend()

ax1.plot(buy_signals.index,
         buy_signals,
         '^', markersize=8, color='green', label='buy')
ax1.plot(sell_signals.index,
         sell_signals,
         'v', markersize=8, color='red', label='sell')
ax1.set_title("Stochastic RSI with 50 & 100 day EMA strategy", fontsize=15)

ax2.plot(all_data.stoch_rsi, color='darkolivegreen', linewidth=0.8)
ax2.axhline(y=20, color='slategrey', linestyle='-')
ax2.axhline(y=80, color='slategrey', linestyle='-')
ax2.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% function to add technical indicators for Strategy 2

def bollinger_bands_with_rsi(df, close_price):

    # RSI - Relative Strength Index
    df['rsi'] = calculate_rsi(close_price, 13)

    # Bollinger bands
    bands = calculate_bollinger_bands(close_price, 30, 2)
    df['upper_bollinger_band'] = bands[0]
    df['lower_bollinger_band'] = bands[1]

    # simple moving average
    df['moving_average_line'] = calculate_sma(close_price, 30)

#%% call the function to get the technical indicators

bollinger_bands_with_rsi(axis_df, axis_df['close'])

#%% Filter out the required columns for this strategy

columns_req = ['close', 'rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band']

data_without_na = axis_df.dropna(subset=['rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band'])

data_subset_2 = data_without_na.loc[:, columns_req]

#%% Create buy and sell signals

data_subset_2.loc[:, 'buy_signal'] = np.where((data_subset_2['rsi'] > 0) & (data_subset_2['rsi'] <= 40) &
                                              (data_subset_2['close'] < data_subset_2['lower_bollinger_band']),
                                              1, 0)

data_subset_2.loc[:, 'sell_signal'] = np.where((data_subset_2['rsi'] >= 60) & (data_subset_2['rsi'] < 100) &
                                               (data_subset_2['close'] > data_subset_2['upper_bollinger_band']),
                                               1, 0)

#%% Filter out points where to buy and sell

buy_sell_data_2 = data_subset_2.loc[(data_subset_2['buy_signal'] == 1) | (data_subset_2['sell_signal'] == 1), :]

buy_sell_data_2[:]['position'] = np.where(buy_sell_data_2['buy_signal'] == 1, 'buy', 'sell')

temp_2 = buy_sell_data_2.groupby((buy_sell_data_2['position'] != buy_sell_data_2['position'].shift()).cumsum()).apply(lambda x: (x.index[0], x.index[-1]))

for tup in temp_2:
    if len(buy_sell_data_2.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_data_2.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_data_2.loc[tup[0]:tup[1], 'close'].idxmin())
            buy_sell_data_2.loc[tup[0]:tup[1], 'position'] = buy_sell_data_2.loc[tup[0]:tup[1], 'position'].where(
                buy_sell_data_2.loc[tup[0]:tup[1], 'position'].index == min_value, 'nothing')

        elif buy_sell_data_2.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_data_2.loc[tup[0]:tup[1], 'close'].idxmax())
            buy_sell_data_2.loc[tup[0]:tup[1], 'position'] = buy_sell_data_2.loc[tup[0]:tup[1], 'position'].where(
                buy_sell_data_2.loc[tup[0]:tup[1], 'position'].index == max_value, 'nothing')


#%% Visualize the data with bollinger bands and rsi

buy_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'buy', 'close']
sell_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'sell', 'close']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_subset_2.close, color='palevioletred', label='Close price', linewidth=1)
ax1.plot(data_subset_2.upper_bollinger_band, color='lightseagreen', label='upper BB', linewidth=1)
ax1.plot(data_subset_2.moving_average_line, color='royalblue', label='MA', linewidth=1)
ax1.plot(data_subset_2.lower_bollinger_band, color='darkorange', label='lower BB', linewidth=1)
ax1.set_xlim([data_subset_2.index[0], data_subset_2.index[-1]])
ax1.legend()

ax1.plot(buy_signals_2.index,
         buy_signals_2,
         '^', markersize=8, color='green', label='buy')
ax1.plot(sell_signals_2.index,
         sell_signals_2,
         'v', markersize=8, color='red', label='sell')
ax1.set_title("Bollinger bands with RSI strategy", fontsize=15)

ax2.plot(data_subset_2.rsi, color='darkolivegreen', linewidth=0.8)
ax2.axhline(y=30, color='slategrey', linestyle='-')
ax2.axhline(y=80, color='slategrey', linestyle='-')
ax2.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% Function to add technical indicators for strategy 3

"""
The MACD line, which is the blue line, in most cases, is usually a 12-day moving average, 
and the signal line, which is the orange line, is usually a 26-day moving average. So, an easy way to figure out if the market is in an uptrend, you simply 
just need to add a 200-day moving average. If we put all this together, we buy if the MACD lines cross below 
the zero line mark, and the current price is also above the 200-day moving average. So as an example, 
we would enter a long trade right here because the MACD lines are crossing upward below the zero line, 
and the current price is above the 200-day moving average. So what you would do is make sure the price 
is above the 200-day moving average.
"""

def macd_with_ema(df, close_price):

    # MACD - Moving Average Convergence Divergence
    macd_signal = calculate_macd(close_price, 12, 26, 9)
    df['macd_line'] = macd_signal[0]
    df['signal_line'] = macd_signal[1]
    df['histogram'] = macd_signal[2]

    # 200-day Exponential moving average
    df['ema_200'] = calculate_ema(close_price, 200)

#%% call the function to get the technical indicators

macd_with_ema(axis_df, axis_df['close'])

#%% Filter out the required columns for this strategy

columns_req = ['close', 'ema_200', 'macd_line', 'signal_line', 'histogram']

data_without_na = axis_df.dropna(subset=['ema_200', 'macd_line', 'signal_line', 'histogram'])

data_subset_3 = data_without_na.loc[:, columns_req]

#%% Create the buy and sell signals

data_subset_3.loc[:, 'diff'] = data_subset_3['macd_line'] - data_subset_3['signal_line']

data_subset_3.loc[:, 'buy_signal'] = np.where((data_subset_3['macd_line'] < 0) &
                                              (data_subset_3['macd_line'] > data_subset_3['signal_line']),
                                              1, 0)

data_subset_3.loc[:, 'sell_signal'] = np.where((data_subset_3['macd_line'] > 0) &
                                               (data_subset_3['macd_line'] < data_subset_3['signal_line']),
                                               1, 0)

#%% Visualize the data with bollinger bands and rsi

# buy_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'buy', 'close']
# sell_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'sell', 'close']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_subset_3.close, color='palevioletred', label='Close price', linewidth=1.3)
ax1.plot(data_subset_3.ema_200, color='royalblue', label='200-day EMA', linewidth=1.3)
ax1.set_xlim([data_subset_3.index[0], data_subset_3.index[2000]])
ax1.legend()

# ax1.plot(buy_signals_2.index,
#          buy_signals_2,
#          '^', markersize=8, color='green', label='buy')
# ax1.plot(sell_signals_2.index,
#          sell_signals_2,
#          'v', markersize=8, color='red', label='sell')
ax1.set_title("Moving Average Convergence Divergence with 200-day EMA strategy", fontsize=15)

ax2.plot(data_subset_3.macd_line, color='dodgerblue', linewidth=0.8)
ax2.plot(data_subset_3.signal_line, color='deeppink', linewidth=0.8)
ax2.axhline(y=0, color='slategrey', linestyle='-')

for i in range(len(data_subset_3['close'])):
    if data_subset_3['histogram'][i] < 0:
        ax2.bar(data_subset_3.index[i], data_subset_3['histogram'][i], color='red')
    else:
        ax2.bar(data_subset_3.index[i], data_subset_3['histogram'][i], color='green')
ax2.set_ylim([-100, 100])


plt.xlabel('Date')
plt.tight_layout()
plt.show()


