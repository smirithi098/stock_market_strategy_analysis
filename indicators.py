import pandas as pd
import numpy as np
class Indicators:

    def __int__(self):
        self.rsi_period = 14
        self.fast_ema_period = 26
        self.slow_ema_period = 12
        self.signal_period = 9

    def calculate_sma(self, price_col, period):
        sma = pd.Series(price_col.transform(lambda x: x.rolling(window=period).mean()))
        return sma

    def calculate_ema(self, price_col, period):
        ema = pd.Series(price_col.ewm(span=period, min_periods=period).mean())
        return ema

    def calculate_rsi(self, price_col):
        # take the first difference of the price column
        price_delta = price_col.diff()

        # Split the series into two: lower close prices & higher close prices
        gain = pd.Series(price_delta.apply(lambda x: x if x > 0 else 0))
        loss = pd.Series(price_delta.apply(lambda x: -x if x < 0 else 0))

        # calculate the EMA for both gains and losses
        gain_ema = self.calculate_ema(gain, self.rsi_period)
        loss_ema = self.calculate_ema(loss, self.rsi_period)

        # RSI = ratio of exponential average gain over a 14-day period  &
        # exponential average loss over a 14-day period
        relative_strength = gain_ema / loss_ema

        rsi = 100 - (100 / (relative_strength + 1))

        return rsi

    def calculate_macd(self, price_col):
        ema_12 = self.calculate_ema(price_col, self.slow_ema_period)
        ema_26 = self.calculate_ema(price_col, self.fast_ema_period)

        macd = ema_26 - ema_12

        signal = self.calculate_ema(price_col, self.signal_period)

        return macd, signal

    def calculate_vwap(self, price_col, vol):
        vwap = ((price_col * vol).cumsum()) / vol.cumsum()

        return vwap
