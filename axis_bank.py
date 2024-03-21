# %% import libraries
import warnings
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
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

#%% Calculate returns

cum_ret_1, portfolio_amount_1 = mod.calculate_cumulative_returns(buy_sell_data_1)

annual_return_1 = mod.calculate_annual_return(buy_sell_data_1, cumulative_return=cum_ret_1)

return_on_investment_1 = mod.calculate_roi(portfolio_amount_1)

print(f"ROI of strategy 1: {return_on_investment_1:.2f}%")
print(f"Annual Return of strategy 1: {annual_return_1 * 100:.2f}%")
print(f"Portfolio Amount at the End of the Investment Period: {portfolio_amount_1:.2f}")

#%% Visualize the cumulative returns over the investment period

fig, ax1 = plt.subplots(nrows=1, figsize=(15, 8))

# Plot the cumulative returns
ax1.plot(buy_sell_data_1.index, buy_sell_data_1['investment_value'], color='darkcyan', label='Portfolio Amount')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Amount')
ax1.tick_params('y')
ax1.grid()
ax1.text(x=0.5, y=1, s='Investment amount for the 50 & 100 day EMA crossover strategy', fontsize=12, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

plt.tight_layout()
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

#%% Calculate returns

cum_ret_2, portfolio_amount_2 = mod.calculate_cumulative_returns(buy_sell_data_2)

annual_return_2 = mod.calculate_annual_return(buy_sell_data_2, cumulative_return=cum_ret_2)

return_on_investment_2 = mod.calculate_roi(portfolio_amount_2)

print(f"ROI of strategy 2: {return_on_investment_2:.2f}%")
print(f"Annual Return of strategy 2: {annual_return_2 * 100:.2f}%")
print(f"Portfolio Amount at the End of the Investment Period: {portfolio_amount_2:.2f}")

#%% Visualize the cumulative returns over the investment period

fig, ax1 = plt.subplots(nrows=1, figsize=(15, 8))

# Plot the cumulative returns
ax1.plot(buy_sell_data_2.index, buy_sell_data_2['investment_value'], color='darkcyan', label='Portfolio Amount')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Amount')
ax1.tick_params('y')
ax1.grid()
ax1.text(x=0.5, y=1, s='Investment amount for the Bollinger bands with RSI strategy', fontsize=12, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

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

#%% Calculate returns

cum_ret_3, portfolio_amount_3 = mod.calculate_cumulative_returns(buy_sell_data_3)

annual_return_3 = mod.calculate_annual_return(buy_sell_data_3, cumulative_return=cum_ret_3)

return_on_investment_3 = mod.calculate_roi(portfolio_amount_3)

print(f"ROI of strategy 2: {return_on_investment_3:.2f}%")
print(f"Annual Return of strategy 2: {annual_return_3 * 100:.2f}%")
print(f"Portfolio Amount at the End of the Investment Period: {portfolio_amount_3:.2f}")

#%% Visualize the cumulative returns over the investment period

fig, ax1 = plt.subplots(nrows=1, figsize=(15, 8))

# Plot the cumulative returns
ax1.plot(buy_sell_data_3.index, buy_sell_data_3['investment_value'], color='darkcyan', label='Portfolio Amount')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Amount')
ax1.tick_params('y')
ax1.grid()
ax1.text(x=0.5, y=1, s='Investment amount for the MACD with 200-day EMA strategy', fontsize=12, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

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

#%% Calculate returns

cum_ret_4, portfolio_amount_4 = mod.calculate_cumulative_returns(buy_sell_data_4)

annual_return_4 = mod.calculate_annual_return(buy_sell_data_4, cumulative_return=cum_ret_4)

return_on_investment_4 = mod.calculate_roi(portfolio_amount_4)

print(f"ROI of strategy 2: {return_on_investment_4:.2f}%")
print(f"Annual Return of strategy 2: {annual_return_4 * 100:.2f}%")
print(f"Portfolio Amount at the End of the Investment Period: {portfolio_amount_4:.2f}")

#%% Visualize the cumulative returns over the investment period

fig, ax1 = plt.subplots(nrows=1, figsize=(15, 8))

# Plot the cumulative returns
ax1.plot(buy_sell_data_4.index, buy_sell_data_4['investment_value'], color='darkcyan', label='Portfolio Amount')
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Amount')
ax1.tick_params('y')
ax1.grid()
ax1.text(x=0.5, y=1, s='Investment amount for the MACD with RSI strategy', fontsize=12, weight='bold',
         ha='center', va='bottom', transform=ax1.transAxes)

plt.tight_layout()
plt.show()

#%% Decision tree Classification Model

from random import randint
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report

#%% Load dataset
df = pd.read_csv('S:/Dissertation 2023/Stock market analysis/stock_market_strategy_analysis/data_files/axis_s1_final.csv')

#%% Model 1: Decision tree without SMOTE analysis - Imbalanced dataset

# Identify predictors and target variable
X_1 = df[['close', 'ema_50', 'ema_100']]
y_1 = df['signal']

# Split data into train and test set
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.3, random_state=789)

# Initialize the Decision Tree Classifier with default parameters
model_1 = DecisionTreeClassifier()

# Fit the model to the training data
model_1.fit(X_train_1, y_train_1)

# Make predictions on the test set
y_pred_1 = model_1.predict(X_test_1)

# Evaluate the model's accuracy
accuracy_1 = accuracy_score(y_test_1, y_pred_1)
print(f"Accuracy for default Decision tree model: {accuracy_1*100:.2f}%")

# Classification report
print("Classification Report - Model 1\n")
print(classification_report(y_test_1, y_pred_1))

# Visualize the confusion matrix
cm_1 = confusion_matrix(y_test_1, y_pred_1)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_1, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("Confusion matrix - Imbalanced data")
plt.show()

#%% Model 2: Decision tree model with SMOTE analysis on both train and test set

# Split the data into buy+50% hold and sell+50% hold
df1 = pd.concat([df[df.signal == 1], df[df.signal == 0].iloc[:3323]])
df2 = pd.concat([df[df.signal == -1], df[df.signal == 0].iloc[:3323]])

# SMOTE Analysis

X1 = df1[['close', 'ema_50', 'ema_100']]
y1 = df1['signal']
sm = SMOTE(random_state=657)
X1_new, y1_new = sm.fit_resample(X1, y1)

X2 = df2[['close', 'ema_50', 'ema_100']]
y2 = df2['signal']
sm = SMOTE()
X2_new, y2_new = sm.fit_resample(X2, y2)

# New dataframes with balanced labels

X_new = pd.concat([X1_new, X2_new])
y_new = pd.concat([y1_new, y2_new])

# Split data into train and test set
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_new, y_new, test_size=0.3, random_state=789)

# Initialize the Decision Tree Classifier with default parameters
model_2 = DecisionTreeClassifier()

# Fit the model to the training data
model_2.fit(X_train_2, y_train_2)

# Make predictions on the test set
y_pred_2 = model_2.predict(X_test_2)

# Evaluate the model's accuracy
accuracy_2 = accuracy_score(y_test_1, y_pred_1)
print(f"Accuracy for Decision tree model with SMOTE: {accuracy_2*100:.2f}%")

# Classification report
print("Classification Report - Model 2\n")
print(classification_report(y_test_2, y_pred_2))

# Visualize the confusion matrix
cm_2 = confusion_matrix(y_test_2, y_pred_2)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm_2, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("Confusion matrix - Balanced data with SMOTE")
plt.show()

