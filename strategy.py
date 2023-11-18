# import libraries
import math
import pandas as pd
import numpy as np
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_ema, calculate_sma

# Function to clean and prepare the initial data loaded from csv file
def prepare_data(df):
    df = df.drop(['series ', '52W H ', '52W L ', 'VALUE ', 'No of trades '],
                 axis=1)
    df = df.rename(columns={'OPEN ': 'open', 'HIGH ': 'high', 'LOW ': 'low', 'PREV. CLOSE ': 'prev_close',
                            'ltp ': 'ltp', 'close ': 'close', 'vwap ': 'vwap', 'VOLUME ': 'volume'})

    df = df.drop_duplicates()
    df.index = pd.to_datetime(df.index)
    df = df.resample('D').asfreq()
    df = df.interpolate('linear')

    return df

# Function to get indicators for strategy 1 - 50 & 100-day EMA crossover
def ema_crossover(df, close_price):

    # EMA - Exponential Moving Average
    df['ema_50'] = calculate_ema(close_price, 50)
    df['ema_100'] = calculate_ema(close_price, 100)

    columns_req = ['close', 'ema_50', 'ema_100']

    df = df.dropna(subset=['ema_50', 'ema_100'])

    df = df.loc[:, columns_req]

    return df

# Function to get indicators for strategy 2 - bollinger bands, moving average line & RSI
def bollinger_bands_with_rsi(df, close_price):
    # RSI - Relative Strength Index
    df['rsi'] = calculate_rsi(close_price, 13)

    # Bollinger bands
    bands = calculate_bollinger_bands(close_price, 30, 2)
    df['upper_bollinger_band'] = bands[0]
    df['lower_bollinger_band'] = bands[1]

    # simple moving average
    df['moving_average_line'] = calculate_sma(close_price, 30)

    columns_req = ['close', 'rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band']

    df = df.dropna(subset=['rsi', 'upper_bollinger_band', 'moving_average_line', 'lower_bollinger_band'])

    df = df.loc[:, columns_req]

    return df

# Function to get indicators for strategy 3 - MACD & 200-day EMA
def macd_with_ema(df, close_price):
    # MACD - Moving Average Convergence Divergence
    macd_signal = calculate_macd(close_price, 12, 26, 9)
    df['macd_line'] = macd_signal[0]
    df['signal_line'] = macd_signal[1]
    df['histogram'] = macd_signal[2]

    # 200-day Exponential moving average
    df['ema_200'] = calculate_ema(close_price, 200)

    columns_req = ['close', 'ema_200', 'macd_line', 'signal_line', 'histogram']

    df = df.dropna(subset=['ema_200', 'macd_line', 'signal_line', 'histogram'])

    df = df.loc[:, columns_req]

    return df

# Function to get indicators for strategy 4 - MACD & RSI
def macd_with_rsi(df, close_price):

    # MACD - Moving Average Convergence Divergence
    macd_signal = calculate_macd(close_price, 12, 26, 9)
    df['macd_line'] = macd_signal[0]
    df['signal_line'] = macd_signal[1]

    # RSI - Relative strength index
    df['rsi'] = calculate_rsi(close_price, 14)

    columns_req = ['close', 'macd_line', 'signal_line', 'rsi']

    df = df.dropna(subset=['macd_line', 'signal_line', 'rsi'])

    df = df.loc[:, columns_req]

    return df

# Function to create buy-sell signals wrt conditions for strategy 1
def get_signals_for_strategy_1(df):
    buy_sell_points = df[(df['position'] != 'nothing')]
    buy_sell_points = buy_sell_points.loc[:buy_sell_points.index[-3], :]

    buy_sell_points = identify_buy_sell_points(buy_sell_points)

    return buy_sell_points

# Function to create buy-sell signals wrt conditions for strategy 2
def get_signals_for_strategy_2(df):
    df.loc[:, 'buy_signal'] = np.where((df['rsi'] > 0) & (df['rsi'] <= 40) &
                                       (df['close'] < df['lower_bollinger_band']),
                                        1, 0)

    df.loc[:, 'sell_signal'] = np.where((df['rsi'] >= 60) & (df['rsi'] < 100) &
                                        (df['close'] > df['upper_bollinger_band']),
                                        1, 0)

    buy_sell_points = df.loc[(df['buy_signal'] == 1) | (df['sell_signal'] == 1), :]

    buy_sell_points.loc[:, 'position'] = np.where(buy_sell_points['buy_signal'] == 1, 'buy', 'sell')

    buy_sell_points = identify_buy_sell_points(buy_sell_points)

    return buy_sell_points

# Function to create buy-sell signals wrt conditions for strategy 3
def get_signals_for_strategy_3(df):
    df.loc[:, 'buy_signal'] = np.where((df['macd_line'] < 0) &
                                       (df['macd_line'] > df['signal_line']),
                                        1, 0)

    df.loc[:, 'sell_signal'] = np.where((df['macd_line'] > 0) &
                                        (df['macd_line'] < df['signal_line']),
                                         1, 0)

    buy_sell_points = df.loc[(df['buy_signal'] == 1) | (df['sell_signal'] == 1), :]

    return buy_sell_points

# Function to identify buy-sell points combining both indicators
def filter_buy_sell_points(df):
    temp = df.groupby((df['position'] != df['position'].shift()).cumsum()).apply(
                        lambda x: (x.index[0], x.index[-1]))

    for tup in temp:
        if len(df.loc[tup[0]:tup[1], :]) > 1:
            if df.loc[tup[0]:tup[1], 'buy_signal'].apply(lambda x: True if x == 1 else False).all():
                if df.loc[tup[0]:tup[1], :].apply(lambda x: True if x.close > x.ema_200 else False, axis=1).all():
                    min_value = pd.to_datetime(df.loc[tup[0]:tup[1], 'close'].idxmin())
                    df.loc[tup[0]:tup[1], 'position'] = np.where(
                        df.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')
                else:
                    df.loc[tup[0]:tup[1], 'position'] = 'nothing'

            elif df.loc[tup[0]:tup[1], 'sell_signal'].apply(lambda x: True if x == 1 else False).all():
                if df.loc[tup[0]:tup[1], :].apply(lambda x: True if x.close < x.ema_200 else False,
                                                               axis=1).all():
                    max_value = pd.to_datetime(df.loc[tup[0]:tup[1], 'close'].idxmax())
                    df.loc[tup[0]:tup[1], 'position'] = np.where(
                        df.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')
                else:
                    df.loc[tup[0]:tup[1], 'position'] = 'nothing'

    return df

# Function to create buy-sell signals wrt conditions for strategy 4
def get_signals_for_strategy_4(df):
    df.loc[:, 'buy_signal'] = np.where((df['rsi'] > 0) & (df['rsi'] <= 40) &
                                       (df['macd_line'] < 0) & (df['macd_line'] > df['signal_line']),
                                        1, 0)

    df.loc[:, 'sell_signal'] = np.where((df['rsi'] >= 60) & (df['rsi'] < 100) &
                                        (df['macd_line'] > 0) & (df['macd_line'] < df['signal_line']),
                                         1, 0)

    buy_sell_points = df.loc[(df['buy_signal'] == 1) | (df['sell_signal'] == 1), :]
    buy_sell_points.loc[:, 'position'] = np.where(buy_sell_points['buy_signal'] == 1, 'buy', 'sell')

    buy_sell_points = identify_buy_sell_points(buy_sell_points)

    return buy_sell_points

# Function to identify the best suitable buy sell points
def identify_buy_sell_points(df):
    temp = df.groupby((df['position'] != df['position'].shift()).cumsum()).apply(
        lambda x: (x.index[0], x.index[-1]))

    for tup in temp:
        if len(df.loc[tup[0]:tup[1], :]) > 1:
            if df.loc[tup[0], 'position'] == 'buy':
                min_value = pd.to_datetime(df.loc[tup[0]:tup[1], 'close'].idxmin())
                df.loc[tup[0]:tup[1], 'position'] = np.where(
                    df.loc[tup[0]:tup[1], 'position'].index == min_value, 'buy', 'nothing')

            elif df.loc[tup[0], 'position'] == 'sell':
                max_value = pd.to_datetime(df.loc[tup[0]:tup[1], 'close'].idxmax())
                df.loc[tup[0]:tup[1], 'position'] = np.where(
                    df.loc[tup[0]:tup[1], 'position'].index == max_value, 'sell', 'nothing')

    df = df[df.position != 'nothing']

    return df

# Function to calculate the capital and returns at every buy-sell points
def calculate_returns(df):
    capital_to_invest = 100000
    df.position = df.position.map({'buy': 1, 'sell': 0})

    for idx in df.index:
        idx_position = df.index.get_loc(idx)

        if df.loc[idx, 'position'] == 1:
            if idx_position == 0:
                df.loc[idx, 'capital'] = capital_to_invest
                df.loc[idx, 'units_bought_or_sold'] = math.floor(df.loc[idx, 'capital'] /
                                                                            df.loc[idx, 'close'])
                df.loc[idx, 'returns'] = df.loc[idx, 'units_bought_or_sold'] * \
                                                    df.loc[idx, 'close']
            else:
                df.loc[idx, 'capital'] = df.loc[df.index[idx_position - 1], 'capital']
                df.loc[idx, 'units_bought_or_sold'] = math.floor(df.loc[idx, 'capital'] /
                                                                            df.loc[idx, 'close'])
                df.loc[idx, 'returns'] = df.loc[idx, 'units_bought_or_sold'] * \
                                                    df.loc[idx, 'close']

        else:
            close_diff = df.loc[idx, 'close'] - df.loc[
                            df.index[idx_position - 1], 'close']
            df.loc[idx, 'units_bought_or_sold'] = df.loc[df.index[idx_position - 1], 'units_bought_or_sold']
            df.loc[idx, 'returns'] = df.loc[idx, 'units_bought_or_sold'] * close_diff
            df.loc[idx, 'capital'] = df.loc[idx, 'returns'] + df.loc[df.index[idx_position - 1], 'returns']

    df.loc[:, '% returns'] = df.loc[:, 'returns'].pct_change()

    returns = get_total_returns(df)

    return returns

# function to calculate the total return given by the strategy
def get_total_returns(ret_df):
    daily_returns = ret_df.loc[:, '% returns'].dropna()

    cumulative_returns = ((1 + daily_returns).cumprod() - 1)

    total = cumulative_returns.loc[cumulative_returns.index[-1]] * 100

    return total
