import pandas as pd
import numpy as np
class Indicators:

    def __int__(self):
        self.rsi_period = 14

    def calculate_sma(self, price_col, period):
        sma = pd.Series(price_col.transform(lambda x: x.rolling(window=period).mean()))
        return sma

    def calculate_ema(self, price_col, period):
        ema = pd.Series(price_col.ewm(span=period, min_periods=period).mean())
        return ema

    def calculate_rsi(self):


    def calculate_macd(self):
        pass

    def calculate_vwap(self):
        pass

