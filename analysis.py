# %% import libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_ema, calculate_sma

# %%  Function - Drop columns, Update column names & Update frequency of index
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


# %%Read the data and prepare

data = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/axisbank.csv",
                   index_col=0, parse_dates=[0], dayfirst=True)

axis_df = data_preparation(data)


# %% function to add technical indicators to df for strategy 1

def ema_crossover(df, close_price):

    # EMA - Exponential Moving Average
    df['ema_50'] = calculate_ema(close_price, 50)
    df['ema_100'] = calculate_ema(close_price, 100)


# %% call the function to get the technical indicators

ema_crossover(axis_df, axis_df['close'])

# %% Filter out rows with value not null

columns_req = ['close', 'ema_50', 'ema_100']

data_without_na = axis_df.dropna(subset=['ema_50', 'ema_100'])

data_subset_1 = data_without_na.loc[:, columns_req]

#%% identify all possible buy-sell points

data_subset_1.loc[:, 'ema_diff'] = data_subset_1['ema_50'] - data_subset_1['ema_100']
data_subset_1.loc[:, 'buy_signal'] = np.where((data_subset_1['ema_50'] > data_subset_1['ema_100']), 1, 0)
data_subset_1.loc[:, 'sell_signal'] = np.where((data_subset_1['ema_50'] < data_subset_1['ema_100']), 1, 0)

#%% Create additional identification columns

crossover_points = pd.DataFrame(data_subset_1.loc[data_subset_1['ema_diff'].between(-1, 1), ['close', 'ema_50', 'ema_diff']])
crossover_points['diff'] = crossover_points['ema_50'].diff()

crossover_points = crossover_points.dropna(axis=0)

#%% Identify points where to buy, sell and hold position

for i, val in enumerate(crossover_points.loc[:, 'diff'].to_list()[:-1]):
    print(i, val)

    if val < 0:
        if (not -10 < val < 10) & (
                val == crossover_points.loc[crossover_points.index[i - 1]:crossover_points.index[i + 2], 'diff'].min()) \
                & (crossover_points.loc[crossover_points.index[i], 'ema_diff'] < crossover_points.loc[crossover_points.index[i + 1], 'ema_diff']):
            crossover_points.loc[crossover_points.index[i], 'position'] = 'buy'
        else:
            crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

    else:
        if (not -10 < val < 10) & (
                val == crossover_points.loc[crossover_points.index[i - 1]:crossover_points.index[i + 2], 'diff'].max()) \
                & (crossover_points.loc[crossover_points.index[i - 1], 'ema_diff'] > crossover_points.loc[crossover_points.index[i], 'ema_diff']):
            crossover_points.loc[crossover_points.index[i], 'position'] = 'sell'
        else:
            crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

# %% Filter out the best suitable buy-sell points for maximum returns

buy_sell_data = crossover_points[(crossover_points['position'] != 'nothing')]
buy_sell_data = buy_sell_data.loc[:buy_sell_data.index[-3], :]

temp = buy_sell_data.groupby((buy_sell_data['position'] != buy_sell_data['position'].shift()).cumsum()).apply(
    lambda x: (x.index[0], x.index[-1]))

for tup in temp:
    if len(buy_sell_data.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_data.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_data.loc[tup[0]:tup[1], 'ema_50'].idxmin())
            buy_sell_data.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')

        elif buy_sell_data.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_data.loc[tup[0]:tup[1], 'ema_50'].idxmax())
            buy_sell_data.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')

# %% visualize the buy sell points with the technical indicators in place

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

#%% Define a function to calculate return percentage

buy_sell_data = buy_sell_data[buy_sell_data.position != 'nothing']
buy_sell_data.position = buy_sell_data.position.map({'buy': 1, 'sell': 0})
capital_to_invest = 100000

for idx in buy_sell_data.index:
    idx_position = buy_sell_data.index.get_loc(idx)

    if buy_sell_data.loc[idx, 'position'] == 1:
        if idx_position == 0:
            buy_sell_data.loc[idx, 'capital'] = capital_to_invest
            buy_sell_data.loc[idx, 'holdings'] = math.floor(buy_sell_data.loc[idx, 'capital'] /
                                                            buy_sell_data.loc[idx, 'close'])
            buy_sell_data.loc[idx, 'returns'] = buy_sell_data.loc[idx, 'holdings'] * \
                                                     buy_sell_data.loc[idx, 'close']
        else:
            buy_sell_data.loc[idx, 'capital'] = buy_sell_data.loc[buy_sell_data.index[idx_position - 1], 'capital']
            buy_sell_data.loc[idx, 'holdings'] = math.floor(buy_sell_data.loc[idx, 'capital'] /
                                                            buy_sell_data.loc[idx, 'close'])
            buy_sell_data.loc[idx, 'returns'] = buy_sell_data.loc[idx, 'holdings'] * \
                                                     buy_sell_data.loc[idx, 'close']

    else:
        close_diff = buy_sell_data.loc[idx, 'close'] - buy_sell_data.loc[buy_sell_data.index[idx_position - 1], 'close']
        buy_sell_data.loc[idx, 'holdings'] = buy_sell_data.loc[buy_sell_data.index[idx_position - 1], 'holdings']
        buy_sell_data.loc[idx, 'returns'] = buy_sell_data.loc[idx, 'holdings'] * buy_sell_data.loc[idx, 'close']
        buy_sell_data.loc[idx, 'capital'] = buy_sell_data.loc[idx, 'returns']

# buy_sell_data = buy_sell_data.drop(['capital'], axis=1)

# %% function to add technical indicators for Strategy 2

def bollinger_bands_with_rsi(df, close_price):
    # RSI - Relative Strength Index
    df['rsi'] = calculate_rsi(close_price, 13)

    # Bollinger bands
    bands = calculate_bollinger_bands(close_price, 30, 2)
    df['upper_bollinger_band'] = bands[0]
    df['lower_bollinger_band'] = bands[1]

    # simple moving average
    df['moving_average_line'] = calculate_sma(close_price, 30)


# %% call the function to get the technical indicators

bollinger_bands_with_rsi(axis_df, axis_df['close'])

# %% Filter out the required columns for this strategy

columns_req = ['close', 'rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band']

data_without_na = axis_df.dropna(subset=['rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band'])

data_subset_2 = data_without_na.loc[:, columns_req]

# %% Create buy and sell signals

data_subset_2.loc[:, 'buy_signal'] = np.where((data_subset_2['rsi'] > 0) & (data_subset_2['rsi'] <= 40) &
                                              (data_subset_2['close'] < data_subset_2['lower_bollinger_band']),
                                              1, 0)

data_subset_2.loc[:, 'sell_signal'] = np.where((data_subset_2['rsi'] >= 60) & (data_subset_2['rsi'] < 100) &
                                               (data_subset_2['close'] > data_subset_2['upper_bollinger_band']),
                                               1, 0)

# %% Filter out points where to buy and sell

buy_sell_data_2 = data_subset_2.loc[(data_subset_2['buy_signal'] == 1) | (data_subset_2['sell_signal'] == 1), :]

buy_sell_data_2[:]['position'] = np.where(buy_sell_data_2['buy_signal'] == 1, 'buy', 'sell')

temp_2 = buy_sell_data_2.groupby((buy_sell_data_2['position'] != buy_sell_data_2['position'].shift()).cumsum()).apply(
    lambda x: (x.index[0], x.index[-1]))

for tup in temp_2:
    if len(buy_sell_data_2.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_data_2.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_data_2.loc[tup[0]:tup[1], 'close'].idxmin())
            buy_sell_data_2.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data_2.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')

        elif buy_sell_data_2.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_data_2.loc[tup[0]:tup[1], 'close'].idxmax())
            buy_sell_data_2.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data_2.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')

# %% Visualize the data with bollinger bands and rsi

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

# %% Function to add technical indicators for strategy 3
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

# %% Create the buy and sell signals

data_subset_3.loc[:, 'diff'] = data_subset_3['macd_line'] - data_subset_3['signal_line']

data_subset_3.loc[:, 'buy_signal'] = np.where((data_subset_3['macd_line'] < 0) &
                                              (data_subset_3['macd_line'] > data_subset_3['signal_line']),
                                              1, 0)

data_subset_3.loc[:, 'sell_signal'] = np.where((data_subset_3['macd_line'] > 0) &
                                               (data_subset_3['macd_line'] < data_subset_3['signal_line']),
                                               1, 0)

#%%

buy_sell_data_3 = data_subset_3.loc[(data_subset_3['buy_signal'] == 1) | (data_subset_3['sell_signal'] == 1), :]
buy_sell_data_3.loc[:, 'position'] = np.where(buy_sell_data_3['buy_signal'] == 1, 'buy', 'sell')
temp_3 = buy_sell_data_3.groupby((buy_sell_data_3['position'] != buy_sell_data_3['position'].shift()).cumsum()).apply(
    lambda x: (x.index[0], x.index[-1]))

for tup in temp_3:
    print(tup[0], tup[1])
    if len(buy_sell_data_3.loc[tup[0]:tup[1], :]) > 1:
        print("length is greater than 1")
        if buy_sell_data_3.loc[tup[0]:tup[1], 'buy_signal'].apply(lambda x: True if x == 1 else False).all():
            print("buy signals")
            if buy_sell_data_3.loc[tup[0]:tup[1], :].apply(lambda x: True if x.close > x.ema_200 else False, axis=1).all():
                print("all points have greater close price than ema")
                min_value = pd.to_datetime(buy_sell_data_3.loc[tup[0]:tup[1], 'close'].idxmin())
                buy_sell_data_3.loc[tup[0]:tup[1], 'position'] = np.where(
                    buy_sell_data_3.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')
            else:
                print("ema is greater")
                buy_sell_data_3.loc[tup[0]:tup[1], 'position'] = 'nothing'

        elif buy_sell_data_3.loc[tup[0]:tup[1], 'sell_signal'].apply(lambda x: True if x == 1 else False).all():
            print("sell signals")
            if buy_sell_data_3.loc[tup[0]:tup[1], :].apply(lambda x: True if x.close < x.ema_200 else False, axis=1).all():
                print("all points have lesser close price than ema")
                max_value = pd.to_datetime(buy_sell_data_3.loc[tup[0]:tup[1], 'close'].idxmax())
                buy_sell_data_3.loc[tup[0]:tup[1], 'position'] = np.where(
                    buy_sell_data_3.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')
            else:
                print("close price is greater")
                buy_sell_data_3.loc[tup[0]:tup[1], 'position'] = 'nothing'


#%% Filter out only the buy-sell points

buy_sell_points = buy_sell_data_3[(buy_sell_data_3['position'] != 'nothing')]

indices = buy_sell_points.groupby((buy_sell_points['position'] != buy_sell_points['position'].shift()).cumsum()).apply(
    lambda x: (x.index[0], x.index[-1]))

for tup in indices:
    if len(buy_sell_points.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_points.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_points.loc[tup[0]:tup[1], 'close'].idxmin())
            buy_sell_points.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_points.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')

        elif buy_sell_points.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_points.loc[tup[0]:tup[1], 'close'].idxmax())
            buy_sell_points.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_points.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')

# %% Visualize the data with MACD and 200-day EMA

buy_signals_3 = buy_sell_points.loc[buy_sell_points['position'] == 'buy', 'close']
sell_signals_3 = buy_sell_points.loc[buy_sell_points['position'] == 'sell', 'close']

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_subset_3.close, color='khaki', label='Close price', linewidth=1.3)
ax1.plot(data_subset_3.ema_200, color='royalblue', label='200-day EMA', linewidth=1.3)
ax1.set_xlim([data_subset_3.index[0], data_subset_3.index[-1]])

ax1.plot(buy_signals_3.index,
         buy_signals_3,
         '^', markersize=8, color='green', label='buy')
ax1.plot(sell_signals_3.index,
         sell_signals_3,
         'v', markersize=8, color='red', label='sell')
ax1.legend()
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

#%% Function to add technical indicators for strategy 4

def macd_with_rsi(df, close_price):

    # MACD - Moving Average Convergence Divergence
    macd_signal = calculate_macd(close_price, 12, 26, 9)
    df['macd_line'] = macd_signal[0]
    df['signal_line'] = macd_signal[1]

    # RSI - Relative strength index
    df['rsi'] = calculate_rsi(close_price, 14)

#%% Call the function to get the required indicators' calculations

macd_with_rsi(axis_df, axis_df['close'])

#%% Filter out the required columns for this strategy

columns_req = ['close', 'macd_line', 'signal_line', 'rsi']

data_without_na = axis_df.dropna(subset=['macd_line', 'signal_line', 'rsi'])

data_subset_4 = data_without_na.loc[:, columns_req]

#%% Identify buy-sell points

data_subset_4.loc[:, 'buy_signal'] = np.where((data_subset_4['rsi'] > 0) & (data_subset_4['rsi'] <= 40) &
                                              (data_subset_4['macd_line'] < 0) &
                                              (data_subset_4['macd_line'] > data_subset_4['signal_line']),
                                              1, 0)

data_subset_4.loc[:, 'sell_signal'] = np.where((data_subset_4['rsi'] >= 60) & (data_subset_4['rsi'] < 100) &
                                               (data_subset_4['macd_line'] > 0) &
                                               (data_subset_4['macd_line'] < data_subset_4['signal_line']),
                                               1, 0)

#%% Create buy-sell positions

buy_sell_data_4 = data_subset_4.loc[(data_subset_4['buy_signal'] == 1) | (data_subset_4['sell_signal'] == 1), :]
buy_sell_data_4.loc[:, 'position'] = np.where(buy_sell_data_4['buy_signal'] == 1, 'buy', 'sell')

#%% Identify single buy-sell points among consecutive points

temp_4 = buy_sell_data_4.groupby((buy_sell_data_4['position'] != buy_sell_data_4['position'].shift()).cumsum()).apply(
    lambda x: (x.index[0], x.index[-1]))

for tup in temp_4:
    if len(buy_sell_data_4.loc[tup[0]:tup[1], :]) > 1:
        if buy_sell_data_4.loc[tup[0], 'position'] == 'buy':
            min_value = pd.to_datetime(buy_sell_data_4.loc[tup[0]:tup[1], 'close'].idxmin())
            buy_sell_data_4.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data_4.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')

        elif buy_sell_data_4.loc[tup[0], 'position'] == 'sell':
            max_value = pd.to_datetime(buy_sell_data_4.loc[tup[0]:tup[1], 'close'].idxmax())
            buy_sell_data_4.loc[tup[0]:tup[1], 'position'] = np.where(
                buy_sell_data_4.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')


#%% Visualize the price with indicators (macd & rsi)

buy_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'buy', 'close']
sell_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'sell', 'close']

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(20, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1, 1]})

ax1.plot(data_subset_4.close, color='darkgoldenrod', label='Close price', linewidth=1.3)
ax1.set_xlim([data_subset_4.index[0], data_subset_4.index[1500]])

ax1.plot(buy_signals_4.index,
         buy_signals_4,
         '^', markersize=8, color='green', label='buy')
ax1.plot(sell_signals_4.index,
         sell_signals_4,
         'v', markersize=8, color='red', label='sell')
ax1.legend()
ax1.set_title("Moving Average Convergence Divergence with RSI strategy", fontsize=15)

ax2.plot(data_subset_4.macd_line, color='dodgerblue', linewidth=0.8)
ax2.plot(data_subset_4.signal_line, color='deeppink', linewidth=0.8)
ax2.axhline(y=0, color='slategrey', linestyle='-')
ax2.set_ylim([-100, 100])

ax3.plot(data_subset_4.rsi, color='darkolivegreen', linewidth=0.8)
ax3.axhline(y=30, color='slategrey', linestyle='-')
ax3.axhline(y=70, color='slategrey', linestyle='-')
ax3.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()
