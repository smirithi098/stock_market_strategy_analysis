# import libraries
import warnings
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from stock_market_strategy_analysis.indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_ema, calculate_sma

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

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

# Function to plot the OHLC - Volume data for the original stock price
def plot_graph(df, symbol):
    pio.renderers.default = 'browser'

    plot = make_subplots(specs=[[{"secondary_y": True}]])

    plot.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Candlestick Graph'
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    plot.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            showlegend=False,
            marker={
                "color": "lightgrey",
            }
        ),
        secondary_y=False,
    )

    for yr in range(df.index[0].year, df.index[-1].year):
        plot.add_vline(x=pd.to_datetime(str(yr) + '-01-01'), line_color='black', line_dash="dash", opacity=0.3)

    plot.update_layout(
        title={
            "text": f"{symbol} Historical Price Data",
            "x": 0.5,
            "y": 0.95
        },
        xaxis_title="Date",
        yaxis_title="Price"
    )

    plot.update_xaxes(
        rangeslider_visible=False,
    )

    plot.update_layout(
        {
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )

    plot.show()

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

# Function to identify points based on condition
def identify_signals(df):
    df.loc[:, 'ema_diff'] = df['ema_50'] - df['ema_100']

    crossover_points = pd.DataFrame(df.loc[df['ema_diff'].between(-1, 1),
                                    ['close', 'ema_50', 'ema_100', 'ema_diff']])
    crossover_points['diff'] = crossover_points['ema_50'].diff()

    crossover_points = crossover_points.dropna(axis=0)

    for i, val in enumerate(crossover_points.loc[:, 'diff'].to_list()[:-2]):
        if val < 0:
            if (not -10 < val < 10) & (val == crossover_points.loc[crossover_points.index[i - 1]:crossover_points.index[i + 2], 'diff'].min()) \
                    & (crossover_points.loc[crossover_points.index[i], 'ema_diff'] < crossover_points.loc[crossover_points.index[i + 1], 'ema_diff']):
                crossover_points.loc[crossover_points.index[i], 'position'] = 'buy'
            else:
                crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

        else:
            if (not -10 < val < 10) & (val == crossover_points.loc[crossover_points.index[i - 1]:crossover_points.index[i + 2], 'diff'].max()) \
                    & (crossover_points.loc[crossover_points.index[i - 1], 'ema_diff'] > crossover_points.loc[crossover_points.index[i], 'ema_diff']):
                crossover_points.loc[crossover_points.index[i], 'position'] = 'sell'
            else:
                crossover_points.loc[crossover_points.index[i], 'position'] = 'nothing'

    return crossover_points

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

    df = df[df.position != 'nothing']

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

    return df

# Function to calculate the capital and returns at every buy-sell points
def calculate_returns(df):
    capital_to_invest = 100000
    df = df[df.position != 'nothing']
    df.position = df.position.map({'buy': 1, 'sell': 0})

    if df.loc[df.index[0], 'position'] == 0:
        df = df.loc[df.index[1]:, :]

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
            if idx_position == 0:
                continue
            else:
                close_diff = df.loc[idx, 'close'] - df.loc[df.index[idx_position - 1], 'close']
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

    cum_return_percent = cumulative_returns * 100

    total = cumulative_returns.loc[cumulative_returns.index[-1]] * 100

    return total, cum_return_percent

# Function to merge all data points
def merge_data(data_subset, position_data):
    df = pd.concat([data_subset, position_data.loc[:, 'position']], axis=1)
    df.position = df.position.fillna('nothing')
    df = df.drop(['buy_signal', 'sell_signal'], axis=1)
    df.position = df.position.map({'buy': 1, 'sell': -1, 'nothing': 0})

    return df

# Function to remove volatility and make TS stationary
def normalize_data(price):
    price_diff = price.diff().dropna()
    price_std = price_diff.groupby([price_diff.index.year, price_diff.index.month]).std()
    price_volatility = price_diff.index.map(lambda dt: price_std.loc[(dt.year, dt.month)])
    price_data = price_diff / price_volatility

    return price_data


# Function to implement the ADABoost classification

def adaboost_classification(df):
    X = df.columns[:-1].to_list()
    y = [df.columns[-1]]

    start_date = df.index[0].date()
    end_date = date(2022, 12, 31)

    diff_in_days = (end_date - start_date).days
    next_date = diff_in_days + 1

    train_start_index = df.index[0]
    train_end_index = df.index[diff_in_days]

    test_start_index = df.index[next_date]

    X_train = df.loc[train_start_index:train_end_index, X]
    X_test = df.loc[test_start_index:, X]

    y_train = df.loc[train_start_index:train_end_index, y]
    y_test = df.loc[test_start_index:, y]

    ab_classifier = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=len(df.columns[1:-1])), n_estimators=750
    )

    ab_classifier.fit(X_train, y_train)

    y_pred = ab_classifier.predict(X_test)

    return y_test, y_pred
