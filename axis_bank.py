# %% import libraries
import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import strategy as mod

sns.set(style='ticks')
warnings.filterwarnings('ignore')
# %% Load and clean the data

data = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/axisbank.csv",
                   index_col=0, parse_dates=[0], dayfirst=True)
# Step 1
axis_df = mod.prepare_data(data)

# %% STRATEGY - 1

# calculate the indicators
data_strat_1 = mod.ema_crossover(axis_df, axis_df['close'])

# %% Identify optimal buy-sell points

buy_sell_data_1 = mod.get_signals_for_strategy_1(data_strat_1.copy(deep=True))

#%% Calculate returns

cum_ret_1, portfolio_amount_1 = mod.calculate_cumulative_returns(buy_sell_data_1)

annual_return_1 = mod.calculate_annual_return(buy_sell_data_1, cumulative_return=cum_ret_1)

print(f"Cumulative Return of strategy 1: {cum_ret_1 * 100:.2f}%")
print(f"Portfolio Amount at the End of the Investment Period: {portfolio_amount_1:.2f}")
print(f"Annual Return of strategy 1: {annual_return_1 * 100:.2f}%")

# %% visualize the buy sell points with the technical indicators in place

buy_signals_1 = buy_sell_data_1.loc[buy_sell_data_1['position'] == 'buy', 'ema_50']
sell_signals_1 = buy_sell_data_1.loc[buy_sell_data_1['position'] == 'sell', 'ema_50']

# Step 6
fig1, ax1 = plt.subplots(nrows=1, figsize=(15, 10))

ax1.plot(buy_sell_data_1.close, color='darkgrey', label='Close price', linewidth=1)
ax1.plot(buy_sell_data_1.ema_50, color='orange', label='50-day EMA', linewidth=1)
ax1.plot(buy_sell_data_1.ema_100, color='blue', label='100-day EMA', linewidth=1)
ax1.set_xlim([buy_sell_data_1.index[0], buy_sell_data_1.index[-1]])

ax1.plot(buy_signals_1.index,
         buy_signals_1,
         '^', markersize=8, color='#116D6E', label='Buy')
ax1.plot(sell_signals_1.index,
         sell_signals_1,
         'v', markersize=8, color='#CD1818', label='Sell')
ax1.legend()

ax1.text(x=0.5, y=1, s='50 & 100 day EMA crossover strategy', fontsize=15, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

plt.xlabel('Date')
plt.tight_layout()
plt.grid()
plt.show()

#%% Visualize the cumulative returns over the investment period

buy_sell_data_1['returns'] = buy_sell_data_1['cumulative_returns'].cumsum()

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)

# Plot the cumulative returns
ax1.plot(buy_sell_data_1.index, buy_sell_data_1['returns'], color='darkcyan', label='Cumulative Returns')
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Returns')
ax1.tick_params('y')

# Plot the portfolio amount
ax2.plot(buy_sell_data_1.index, buy_sell_data_1['investment_value'], color='indianred', label='Portfolio Amount')
ax2.set_ylabel('Portfolio Amount')
ax2.tick_params('y')

# Add a legend
fig.legend(loc='upper right')

# Show the plot
plt.show()


#%% Plot the portfolio value over the investment period

sns.lineplot(data=buy_sell_data_1, x=buy_sell_data_1.index, y='investment_value', color='blue')

# Set the x-axis label
plt.xlabel('Date')

# Set the y-axis label
plt.ylabel('Portfolio Amount')

# Set the title of the plot
plt.title('Investment Value Over Time')

# Display the plot
plt.show()

#%% STRATEGY 2
# Get the technical indicators for the strategy 2

# Step 1
data_strat_2 = mod.bollinger_bands_with_rsi(axis_df, axis_df['close'])

#%% Identify the buy-sell points

buy_sell_data_2 = mod.get_signals_for_strategy_2(data_strat_2.copy(deep=True))


#%% visualize the buy sell points with the technical indicators in place

buy_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['signal'] == 1, 'lower_bollinger_band']
sell_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['signal'] == -1, 'upper_bollinger_band']

fig2, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(buy_sell_data_2.close, color='blue', label='Close price', linewidth=1)
ax1.plot(buy_sell_data_2.upper_bollinger_band, color='orange', label='upper BB', linewidth=1)
ax1.plot(buy_sell_data_2.moving_average_line, color='darkgrey', label='MA', linewidth=1)
ax1.plot(buy_sell_data_2.lower_bollinger_band, color='orange', label='lower BB', linewidth=1)
ax1.set_xlim([buy_sell_data_2.index[0], buy_sell_data_2.index[-1]])

ax1.plot(buy_signals_2.index,
         buy_signals_2,
         '^', markersize=8, color='#116D6E', label='buy')
ax1.plot(sell_signals_2.index,
         sell_signals_2,
         'v', markersize=8, color='#CD1818', label='sell')
ax1.legend()
ax1.text(x=0.5, y=1, s='Bollinger bands with RSI strategy', fontsize=15, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

ax1.grid()

ax2.plot(buy_sell_data_2.rsi, color='darkolivegreen', linewidth=0.8)
ax2.axhline(y=30, color='slategrey', linestyle='-')
ax2.axhline(y=70, color='slategrey', linestyle='-')
ax2.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% STRATEGY - 3

# Get the technical indicators for the strategy 3

data_strat_3 = mod.macd_with_ema(axis_df, axis_df['close'])

#%% Identify buy-sell points

buy_sell_data_3 = mod.get_signals_for_strategy_3(data_strat_3.copy(deep=True))

#%% Visualize the indicators with the identified buy-sell points

buy_signals_3 = buy_sell_data_3.loc[buy_sell_data_3['position'] == 'buy', 'close']
sell_signals_3 = buy_sell_data_3.loc[buy_sell_data_3['position'] == 'sell', 'close']

fig3, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(buy_sell_data_3.close, color='grey', label='Close price', linewidth=1.3)
ax1.plot(buy_sell_data_3.ema_200, color='#C84B31', label='200-day EMA', linewidth=1.3)
ax1.set_xlim([buy_sell_data_3.index[0], buy_sell_data_3.index[-1]])

ax1.plot(buy_signals_3.index,
         buy_signals_3,
         '^', markersize=8, color='#116D6E', label='buy')
ax1.plot(sell_signals_3.index,
         sell_signals_3,
         'v', markersize=8, color='#CD1818', label='sell')
ax1.legend()
ax1.text(x=0.5, y=1, s='Moving Average Convergence Divergence with 200-day EMA strategy', fontsize=15, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

ax1.grid()

ax2.plot(buy_sell_data_3.macd_line, color='dodgerblue', linewidth=0.8)
ax2.plot(buy_sell_data_3.signal_line, color='deeppink', linewidth=0.8)
ax2.axhline(y=0, color='slategrey', linestyle='-')

ax2.set_ylim([-100, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% STRATEGY - 4

# Get the technical indicators for the strategy 4

data_strat_4 = mod.macd_with_rsi(axis_df, axis_df['close'])

#%% Get the buy-sell points

buy_sell_data_4 = mod.get_signals_for_strategy_4(data_strat_4.copy(deep=True))

#%% Visualize the price with indicators (macd & rsi)

buy_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'buy', 'close']
sell_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'sell', 'close']

fig4, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(20, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1, 1]})

ax1.plot(buy_sell_data_4.close, color='grey', label='Close price', linewidth=1.3)
ax1.set_xlim([buy_sell_data_4.index[0], buy_sell_data_4.index[-1]])

ax1.plot(buy_signals_4.index,
         buy_signals_4,
         '^', markersize=8, color='#116D6E', label='buy')
ax1.plot(sell_signals_4.index,
         sell_signals_4,
         'v', markersize=8, color='#CD1818', label='sell')
ax1.legend()
ax1.text(x=0.5, y=1, s='Moving Average Convergence Divergence with RSI strategy', fontsize=15, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

ax1.grid()

ax2.plot(buy_sell_data_4.macd_line, color='dodgerblue', linewidth=0.8)
ax2.plot(buy_sell_data_4.signal_line, color='deeppink', linewidth=0.8)
ax2.axhline(y=0, color='slategrey', linestyle='-')
ax2.set_ylim([-100, 100])

ax3.plot(buy_sell_data_4.rsi, color='darkolivegreen', linewidth=0.8)
ax3.axhline(y=30, color='slategrey', linestyle='-')
ax3.axhline(y=70, color='slategrey', linestyle='-')
ax3.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

