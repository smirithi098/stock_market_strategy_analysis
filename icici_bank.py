# %% import libraries
import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import stock_market_strategy_analysis.strategy as strat

sns.set(style='ticks')
warnings.filterwarnings('ignore')
#%% Load and clean the data

data = pd.read_csv("S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/icicibank.csv",
                   index_col=0, parse_dates=[0], dayfirst=True)
# Step 1
icici_df = strat.prepare_data(data)

#%% Visualize the ICICI BANK OHLC prices with volume

strat.plot_graph(icici_df, 'ICICI BANK')

# ##########################################  STRATEGY - 1  ############################################################
#%% Get the technical indicators for the strategy 1

# Step 2
data_subset_1 = strat.ema_crossover(icici_df, icici_df['close'])

#%% identify all possible buy-sell points

# Step 3
crossover_points = strat.identify_signals(data_subset_1)

#%% Get the best suited buy-sell points

# Step 4
buy_sell_data = strat.get_signals_for_strategy_1(crossover_points)

buy_signals = buy_sell_data.loc[buy_sell_data['position'] == 'buy', 'close']
sell_signals = buy_sell_data.loc[buy_sell_data['position'] == 'sell', 'close']

#%% Get the total return % given by the strategy

# Step 5
total_return = strat.calculate_returns(buy_sell_data)

# %% visualize the buy sell points with the technical indicators in place

# Step 6
fig, ax1 = plt.subplots(nrows=1, figsize=(15, 10))

ax1.plot(data_subset_1.close, color='#7F8487', label='Close price', linewidth=1)
ax1.plot(data_subset_1.ema_50, color='#704F4F', label='50-day EMA', linewidth=1)
ax1.plot(data_subset_1.ema_100, color='#AF0171', label='100-day EMA', linewidth=1)
ax1.set_xlim([data_subset_1.index[0], data_subset_1.index[-1]])

ax1.plot(buy_signals.index,
         buy_signals,
         '^', markersize=8, color='#116D6E', label='buy')
ax1.plot(sell_signals.index,
         sell_signals,
         'v', markersize=8, color='#CD1818', label='sell')
ax1.legend()

ax1.text(x=0.5, y=1, s='50 & 100 day EMA crossover strategy', fontsize=15, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)


plt.xlabel('Date')
plt.tight_layout()
plt.grid()
plt.show()

#%% Get the final data with all points merged with buy-sell-hold position

merged_data = strat.merge_data(data_subset_1, buy_sell_data)

#%% Normalize the columns to have equal scales

for column in merged_data.columns[:-1]:
    merged_data[column] = strat.normalize_data(merged_data[column])

merged_data = merged_data.dropna()

#%% Call the function to train and test the classifier

y_tuple = strat.adaboost_classification(merged_data)
y_actual = y_tuple[0]
y_predicted = y_tuple[1]

# plot the confusion matrix

conf_mat = confusion_matrix(y_actual, y_predicted)

sns.heatmap(conf_mat, annot=True, cmap='Greens')
plt.title("Confusion Matrix For EMA crossover strategy")
plt.tight_layout()
plt.show()

# ##########################################  STRATEGY - 2  ############################################################
#%% Get the technical indicators for the strategy 2

# Step 2
data_subset_2 = strat.bollinger_bands_with_rsi(icici_df, icici_df['close'])

#%% Get the buy-sell points

# Step 3
buy_sell_data_2 = strat.get_signals_for_strategy_2(data_subset_2)

buy_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'buy', 'close']
sell_signals_2 = buy_sell_data_2.loc[buy_sell_data_2['position'] == 'sell', 'close']

#%% Get the total return % given by the strategy

# Step 4
total_return_2 = strat.calculate_returns(buy_sell_data_2)

#%% Visualize the data with bollinger bands and rsi

# Step 5

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_subset_2.close, color='#7F8487', label='Close price', linewidth=1)
ax1.plot(data_subset_2.upper_bollinger_band, color='#0E8388', label='upper BB', linewidth=1)
ax1.plot(data_subset_2.moving_average_line, color='#AF0171', label='MA', linewidth=1)
ax1.plot(data_subset_2.lower_bollinger_band, color='#C84B31', label='lower BB', linewidth=1)
ax1.set_xlim([data_subset_2.index[0], data_subset_2.index[-1]])

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

ax2.plot(data_subset_2.rsi, color='darkolivegreen', linewidth=0.8)
ax2.axhline(y=30, color='slategrey', linestyle='-')
ax2.axhline(y=80, color='slategrey', linestyle='-')
ax2.set_ylim([0, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% Get the final data with all points merged with buy-sell-hold position

merged_data = strat.merge_data(data_subset_2, buy_sell_data_2)

#%% Normalize the columns to have equal scales

for column in merged_data.columns[:-1]:
    merged_data[column] = strat.normalize_data(merged_data[column])

merged_data = merged_data.dropna()

#%% Call the function to train and test the classifier

y_tuple = strat.adaboost_classification(merged_data)
y_actual = y_tuple[0]
y_predicted = y_tuple[1]

# plot the confusion matrix

conf_mat = confusion_matrix(y_actual, y_predicted)

sns.heatmap(conf_mat, annot=True, cmap='Greens')
plt.title("Confusion Matrix For Bollinger bands with RSI strategy")
plt.tight_layout()
plt.show()

# ##########################################  STRATEGY - 3  ############################################################

#%% Get the technical indicators for the strategy 3

# Step 2
data_subset_3 = strat.macd_with_ema(icici_df, icici_df['close'])

#%% Get the buy-sell points

# Step 3
buy_sell_points_3 = strat.get_signals_for_strategy_3(data_subset_3)

buy_sell_points_3.loc[:, 'position'] = np.where(buy_sell_points_3['buy_signal'] == 1, 'buy', 'sell')

buy_sell_points = strat.filter_buy_sell_points(buy_sell_points_3)

buy_sell_data_3 = strat.identify_buy_sell_points(buy_sell_points)

buy_signals_3 = buy_sell_data_3.loc[buy_sell_data_3['position'] == 'buy', 'close']
sell_signals_3 = buy_sell_data_3.loc[buy_sell_data_3['position'] == 'sell', 'close']

#%% Get the total return % given by the strategy

# Step 4

total_return_3 = strat.calculate_returns(buy_sell_data_3)

# %% Visualize the data with MACD and 200-day EMA

# Step 5

fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20, 10), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(data_subset_3.close, color='darkgrey', label='Close price', linewidth=1.3)
ax1.plot(data_subset_3.ema_200, color='#C84B31', label='200-day EMA', linewidth=1.3)
ax1.set_xlim([data_subset_3.index[0], data_subset_3.index[-1]])

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

ax2.plot(data_subset_3.macd_line, color='dodgerblue', linewidth=0.8)
ax2.plot(data_subset_3.signal_line, color='deeppink', linewidth=0.8)
ax2.axhline(y=0, color='slategrey', linestyle='-')

ax2.set_ylim([-100, 100])

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%% Get the final data with all points merged with buy-sell-hold position

merged_data = strat.merge_data(data_subset_3, buy_sell_data_3)

#%% Normalize the columns to have equal scales

for column in merged_data.columns[:-1]:
    merged_data[column] = strat.normalize_data(merged_data[column])

merged_data = merged_data.dropna()

#%% Call the function to train and test the classifier

y_tuple = strat.adaboost_classification(merged_data)
y_actual = y_tuple[0]
y_predicted = y_tuple[1]

# plot the confusion matrix

conf_mat = confusion_matrix(y_actual, y_predicted)

sns.heatmap(conf_mat, annot=True, cmap='Greens')
plt.title("Confusion Matrix For MACD with EMA strategy")
plt.tight_layout()
plt.show()


# ##########################################  STRATEGY - 4  ############################################################

#%% Get the technical indicators for the strategy 4

# Step 2
data_subset_4 = strat.macd_with_rsi(icici_df, icici_df['close'])

#%% Get the buy-sell points

# Step 3
buy_sell_data_4 = strat.get_signals_for_strategy_4(data_subset_4)

buy_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'buy', 'close']
sell_signals_4 = buy_sell_data_4.loc[buy_sell_data_4['position'] == 'sell', 'close']

#%% Get the total return % given by the strategy

# Step 4

total_return_4 = strat.calculate_returns(buy_sell_data_4)

#%% Visualize the price with indicators (macd & rsi)

# Step 5

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(20, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1, 1]})

ax1.plot(data_subset_4.close, color='#4A55A2', label='Close price', linewidth=1.3)
ax1.set_xlim([data_subset_4.index[0], data_subset_4.index[1500]])

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

#%% Get the final data with all points merged with buy-sell-hold position

merged_data = strat.merge_data(data_subset_4, buy_sell_data_4)

#%% Normalize the columns to have equal scales

for column in merged_data.columns[:-1]:
    merged_data[column] = strat.normalize_data(merged_data[column])

merged_data = merged_data.dropna()

#%% Call the function to train and test the classifier

y_tuple = strat.adaboost_classification(merged_data)
y_actual = y_tuple[0]
y_predicted = y_tuple[1]

# plot the confusion matrix

conf_mat = confusion_matrix(y_actual, y_predicted)

sns.heatmap(conf_mat, annot=True, cmap='Greens')
plt.title("Confusion Matrix For MACD with RSI strategy")
plt.tight_layout()
plt.show()
