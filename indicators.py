import pandas as pd
import numpy as np

def calculate_sma(price_col, period):
    sma = pd.Series(price_col.transform(lambda x: x.rolling(window=period).mean()))
    return sma

def calculate_std_ma(price_col, period):
    std_ma = pd.Series(price_col.transform(lambda x: x.rolling(window=period).std()))
    return std_ma

def calculate_ema(price_col, period):
    ema = pd.Series(price_col.ewm(span=period, min_periods=period).mean())
    return ema

def calculate_rsi(price_col, rsi_period):
    # take the first difference of the price column
    price_delta = price_col.diff()

    # Split the series into two: lower close prices & higher close prices
    gain = pd.Series(price_delta.apply(lambda x: x if x > 0 else 0))
    loss = pd.Series(price_delta.apply(lambda x: -x if x < 0 else 0))

    # calculate the EMA for both gains and losses
    gain_ema = calculate_ema(gain, rsi_period)
    loss_ema = calculate_ema(loss, rsi_period)

    # RSI = ratio of exponential average gain over a 14-day period  &
    # exponential average loss over a 14-day period
    relative_strength = gain_ema / loss_ema

    rsi = 100 - (100 / (relative_strength + 1))

    return rsi

def calculate_macd(price_col, slow_period, fast_period, signal_period):
    ema_12 = calculate_ema(price_col, slow_period)
    ema_26 = calculate_ema(price_col, fast_period)

    macd = ema_12 - ema_26

    signal = calculate_ema(macd, signal_period)

    return macd, signal

def calculate_bollinger_bands(price_col, ma_period, std_value):
    ma_line = calculate_sma(price_col, ma_period)
    std_line = calculate_std_ma(price_col, ma_period)

    upper_band = ma_line + (std_value * std_line)
    lower_band = ma_line - (std_value * std_line)

    return upper_band, lower_band

def calculate_stochastic_rsi(price_col, period):
    # take the first difference of the price column
    price_delta = price_col.diff()

    # Split the series into two: lower close prices & higher close prices
    gain = pd.Series(price_delta.apply(lambda x: x if x > 0 else 0))
    loss = pd.Series(price_delta.apply(lambda x: -x if x < 0 else 0))

    period_high = gain.rolling(period).max()
    period_low = loss.rolling(period).min()

    stoch_rsi = (price_col - period_low) / (period_high - period_low)

    return stoch_rsi

