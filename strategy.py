# import libraries
import warnings
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from indicators import calculate_rsi, \
    calculate_macd, calculate_bollinger_bands, calculate_ema, calculate_sma

from sklearn.model_selection import train_test_split
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

# Function to create buy-sell signals wrt conditions for strategy 1
def get_signals_for_strategy_1(df):
    df['signal'] = 0
    df['position'] = None

    for n in range(1, len(df)):
        if df['ema_50'][n] > df['ema_100'][n] and df['ema_50'][n - 1] <= df['ema_100'][n - 1]:
            df['signal'][n] = 1
            df['position'][n] = 'buy'

        elif df['ema_50'][n] < df['ema_100'][n] and df['ema_50'][n - 1] >= df['ema_100'][n - 1]:
            df['signal'][n] = -1
            df['position'][n] = 'sell'
        else:
            df['signal'][n] = 0
            df['position'][n] = 'hold'

    return df

# Function to create buy-sell signals wrt conditions for strategy 2
def get_signals_for_strategy_2(df):
    df['signal'] = 0
    df['position'] = None
    trade_position = False

    for n in range(1, len(df)):
        if (df['rsi'][n] > 0) and (df['rsi'][n] <= 30) and (df['close'][n] < df['lower_bollinger_band'][n]) \
                and (df['close'][n - 1] >= df['lower_bollinger_band'][n - 1]) and not trade_position:
            df['signal'][n] = 1
            df['position'][n] = 'buy'
            trade_position = True

        elif (df['rsi'][n] >= 70) and (df['rsi'][n] < 100) and (df['close'][n] > df['upper_bollinger_band'][n]) \
                and (df['close'][n - 1] <= df['upper_bollinger_band'][n - 1]) and trade_position:
            df['signal'][n] = -1
            df['position'][n] = 'sell'
            trade_position = False

        else:
            df['signal'][n] = 0
            df['position'][n] = 'hold'

    return df

# Function to create buy-sell signals wrt conditions for strategy 3
def get_signals_for_strategy_3(df):
    df['signal'] = 0
    df['position'] = None
    trade_position = False

    for n in range(1, len(df)):
        if (df['macd_line'][n] < 0) and (df['signal_line'][n] < 0) and (df['macd_line'][n] > df['signal_line'][n]) and \
                (df['close'][n] > df['ema_200'][n]) and (df['close'][n - 1] <= df['ema_200'][n - 1]) \
                and not trade_position:
            df['signal'][n] = 1
            df['position'][n] = 'buy'
            trade_position = True

        elif (df['macd_line'][n] > 0) and (df['signal_line'][n] > 0) and (df['macd_line'][n] < df['signal_line'][n]) and \
                (df['close'][n] < df['ema_200'][n]) and (df['close'][n - 1] >= df['ema_200'][n - 1]) and \
                trade_position:
            df['signal'][n] = -1
            df['position'][n] = 'sell'
            trade_position = False

        else:
            df['signal'][n] = 0
            df['position'][n] = 'hold'

    return df

# Function to create buy-sell signals wrt conditions for strategy 4
def get_signals_for_strategy_4(df):
    df['signal'] = 0
    df['position'] = None
    trade_position = False

    for n in range(1, len(df)):
        if (df['rsi'][n] > 0) and (df['rsi'][n] <= 30) and (df['macd_line'][n] < 0) and (df['signal_line'][n] < 0) and \
                (df['macd_line'][n] > df['signal_line'][n]) and not trade_position:
            df['signal'][n] = 1
            df['position'][n] = 'buy'
            trade_position = True

        elif (df['rsi'][n] >= 70) and (df['rsi'][n] < 100) and (df['macd_line'][n] > 0) and (df['signal_line'][n] > 0) and \
                (df['macd_line'][n] < df['signal_line'][n]) and trade_position:
            df['signal'][n] = -1
            df['position'][n] = 'sell'
            trade_position = False

        else:
            df['signal'][n] = 0
            df['position'][n] = 'hold'

    return df

# Function to calculate the capital and returns at every buy-sell points
def calculate_returns(df):
    df['daily_returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['daily_returns'] * df['signal'].shift(1)
    df['cum_returns'] = (df['strategy_returns'] + 1).cumprod()

    initial_amount_invested = 10000

    df['portfolio_amount'] = initial_amount_invested * df['cum_returns']
    df['perc'] = 0.2

    start_year = df.index[0].year
    current_year = df.index[-1].year
    total_years = current_year - start_year

    annual_return = (1 + df['cum_returns'].iloc[-1]) ** (1/total_years) - 1
    portfolio_amount = df['portfolio_amount'].iloc[-1]

    return annual_return, portfolio_amount

def calculate_cumulative_returns(df):
    current_signal = None
    price_bought = 0
    investment_amount = 10000
    df['cumulative_returns'] = 0
    df['investment_value'] = 0

    last_generated_value = 0

    for i in range(1, len(df)):
        if df['position'][i] == 'buy':
            price_bought = df['close'][i]
            current_signal = 1
            df['investment_value'][i] = investment_amount
            last_generated_value = investment_amount

        elif df['position'][i] == 'sell' and current_signal == 1:
            percentage_return = (df['close'][i] - price_bought) / price_bought
            df['cumulative_returns'][i] = percentage_return
            investment_amount *= (1 + percentage_return)
            df['investment_value'][i] = investment_amount
            last_generated_value = investment_amount
            current_signal = None

        elif df['position'][i] == 'hold':
            df['investment_value'][i] = last_generated_value

    cumulative_return = (1 + df['cumulative_returns'].sum()) - 1

    return cumulative_return, investment_amount

def calculate_annual_return(df, cumulative_return):
    date_diff = relativedelta(df.index[0], df.index[-1])
    total_years = date_diff.years + \
                  date_diff.months / 12 + \
                  date_diff.days / 365.25

    annual_return = (1 + cumulative_return)**(1/total_years) - 1

    return annual_return

def calculate_roi(investment_amount):
    initial_amount = 10000
    final_amount = investment_amount

    roi = ((final_amount - initial_amount) / initial_amount) * 100

    return roi
