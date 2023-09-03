import matplotlib.pyplot as plt
import requests
import pandas as pd
from io import StringIO
import numpy as np


symbol = "META"
apikey = "UUR6RZ7V0UP53UHN"  

# Use the TIME_SERIES_DAILY function with outputsize set to 'full' to get up to 20 years of daily data
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={apikey}'

response = requests.get(url)
data = response.json()

# Extract time series data
long_data = pd.DataFrame(data['Time Series (Daily)']).T

# Convert the index to datetime format
long_data.index = pd.to_datetime(long_data.index)

# Convert the stock data to floats
long_data = long_data.astype(float)

# Reverse the DataFrame to chronological order
long_data = long_data.iloc[::-1]

# Filter the last 2 years of data
long_data = long_data.loc[long_data.index.year >= pd.Timestamp.today().year - 2]

# Rename columns (assuming that the column names are the same as in the provided documentation)
long_data.columns = ['open', 'high', 'low', 'close', 'volume']

# Reset index
long_data.reset_index(drop=True, inplace=True)

# Create a new DataFrame for hourly data
hourly_data = pd.DataFrame()

new_rows = []

for i in range(len(long_data) - 1):
    for j in range(7):
        # Simple average of the current and next row
        new_row = 0.5 * (long_data.iloc[i] + long_data.iloc[i + 1])

        # For the close price, generate a random value between the previous and current close
        previous_close = long_data.iloc[i]['close']
        current_close = long_data.iloc[i + 1]['close']

        # Allow up to 1% increase beyond the current close
        upper_limit = current_close * 1.01

        random_close = np.random.uniform(low=previous_close, high=upper_limit)
        new_row['close'] = random_close

        new_rows.append(new_row.to_dict())

hourly_data = pd.DataFrame(new_rows)

print(hourly_data)
long_data=hourly_data








#33k combinations in 7min
#adjust the indicator values
# Define the parameter space
sma_length = np.arange(10, 90, 20) #4
sma_long = np.arange(100, 250, 50) # 4
standard_deviation = np.arange(1.2, 1.8, 0.1) # 7
tsi_length= np.arange (25, 100, 25) # 4
ROC = np.arange(25, 100, 25) #4
SMA_direction_raw_number = np.arange(0.005, 0.025, 0.005) # 5
TSI_min = np.arange(0.1, 0.25, 0.05) # 4





# Generate the Cartesian product of the parameter space
parameter_space = list(itertools.product(sma_length, sma_long, standard_deviation, tsi_length, ROC, SMA_direction_raw_number, TSI_min))

# Initialize a np array to store the results
loop_results = []

# Calculate the technical indicators once outside the loop


def calculate_rsi(data, window):
    delta = data.diff()
    loss = delta.where(delta < 0, 0)
    gain = -delta.where(delta > 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_tsi(data, r, s, signal_period):
    """
    Calculate True Strength Index (TSI) and its Signal line
    :param data: DataFrame
    :param r: int
        The time period for calculating momentum (default is 25)
    :param s: int
        The time period for calculating smoothed moving averages (default is 13)
    :param signal_period: int
        The time period for calculating the Signal line (default is 9)
    :return: DataFrame
    """
    diff = data.diff(1)
    diff.fillna(0, inplace=True)

    # Calculate absolute diff
    abs_diff = abs(diff)

    # Calculate EMA of diff
    EMA_diff = diff.ewm(span=r).mean()

    # Calculate EMA of abs_diff
    EMA_abs_diff = abs_diff.ewm(span=r).mean()

    # Calculate EMA of EMA_diff
    EMA_EMA_diff = EMA_diff.ewm(span=s).mean()

    # Calculate EMA of EMA_abs_diff
    EMA_EMA_abs_diff = EMA_abs_diff.ewm(span=s).mean()

    # Calculate TSI
    TSI = pd.Series(EMA_EMA_diff / EMA_EMA_abs_diff, name='TSI')

    # Calculate Signal line
    Signal = TSI.rolling(window=signal_period).mean()

    return TSI, Signal

# Loop over the parameter space
for sma_length, sma_long, standard_deviation, tsi_length, ROC, SMA_direction_raw_number, TSI_min in parameter_space:
    rsi_length = tsi_length
    # Update the technical indicators for the current parameters
    technical_indicators['SMA'] = long_data['close'].rolling(sma_length).mean()
    technical_indicators["SMA_long"] = long_data["close"].rolling(sma_long).mean()
    technical_indicators['STD'] = long_data['close'].rolling(sma_length).std()
    technical_indicators['upper_band'] = technical_indicators['SMA'] + (technical_indicators['STD'] * standard_deviation)
    technical_indicators['lower_band'] = technical_indicators['SMA'] - (technical_indicators['STD'] * standard_deviation)




# Assuming 'close' is your pandas Series with closing prices


# Calculate TSI and Signal line
    tsi_line, signal = calculate_tsi(technical_indicators["price"], tsi_length, int(tsi_length/2), int(tsi_length/3))

    # Assign TSI and Signal to the technical_indicators DataFrame
    technical_indicators['TSI'] = tsi_line
    technical_indicators['TSI_signal'] = signal








#Enter your BUY logic here and declare buy_condition


  
#Enter your SELL logic here and declare sell_condition








    buy_price = [None]*len(technical_indicators)
    sell_price = [None]*len(technical_indicators)
    cash_available=True


    for i in range(len(technical_indicators)):
        if cash_available and buy_condition.iloc[i]:
            buy_price[i] = technical_indicators['price'].iloc[i]
            cash_available = False
        elif not cash_available and sell_condition.iloc[i]:
            sell_price[i] = technical_indicators['price'].iloc[i]
            cash_available = True



    technical_indicators['buy_price'] = buy_price
    technical_indicators['sell_price'] = sell_price




    # Initialize a list for completed trades
    trades = []

    # Initialize variables to hold the last buy price and date
    last_buy_price = None
    last_buy_date = None

    # Loop through the DataFrame
    for i in range(len(technical_indicators)):
        # If this row is a buy signal, update the last buy price and date
        if not pd.isnull(technical_indicators['buy_price'].iloc[i]):
            last_buy_price = technical_indicators['buy_price'].iloc[i]
            last_buy_date = technical_indicators.index[i]
        # If this row is a sell signal and there was a previous buy signal, record the trade
        elif not pd.isnull(technical_indicators['sell_price'].iloc[i]) and last_buy_price is not None:
            last_sell_price = technical_indicators['sell_price'].iloc[i]
            sell_date = technical_indicators.index[i]
            profit = last_sell_price - last_buy_price
            hold_period = (sell_date - last_buy_date)/7
            trade = {'buy_date': last_buy_date, 'buy_price': last_buy_price, 'sell_date': sell_date, 'sell_price': last_sell_price, 'profit': profit, 'hold_period': hold_period}
            trades.append(trade)
            last_buy_price = None
            last_buy_date = None


    # New logic
    # If there was a recent buy that hasn't been sold yet
    if last_buy_price is not None:
        last_sell_price = technical_indicators['price'].iloc[-1]  # Last available price
        sell_date = technical_indicators.index[-1]
        profit = last_sell_price - last_buy_price
        hold_period = (sell_date - last_buy_date)/7
        trade = {'buy_date': last_buy_date, 'buy_price': last_buy_price, 'sell_date': sell_date, 'sell_price': last_sell_price, 'profit': profit, 'hold_period': hold_period}
        trades.append(trade)

    # Convert the list of trades to a DataFrame
    trades = pd.DataFrame(trades)



    if not trades.empty:
        trades["profit%%%"] = trades["profit"] * 100 / trades["buy_price"]

        profit = trades["profit"].sum()
        profit_percentage = trades["profit%%%"].sum()
        nr_of_trades = len(trades)
        average_hold_period = trades["hold_period"].mean()
        nr_of_successful_trades = (trades["profit%%%"] > 0).sum()
        nr_of_optimal_trades = (trades["profit%%%"] > 1).sum()

        if nr_of_trades == 0:
            efficiency = 0
            optimal_efficiency = 0
        else:
            efficiency = nr_of_successful_trades / nr_of_trades
            optimal_efficiency = nr_of_optimal_trades / nr_of_trades

    else:
        profit = 0
        profit_percentage = 0
        nr_of_trades = 0
        average_hold_period = 0
        efficiency = 0
        optimal_efficiency = 0


    loop = {'sma_length': sma_length, "sma_long": sma_long, "standard_deviation": standard_deviation, "tsi_length": tsi_length, "ROC": ROC, "SMA_direction_raw_number": SMA_direction_raw_number, "TSI_min": TSI_min, 'profit': profit, "profit_percentage": profit_percentage, 'average_hold_period': average_hold_period, "nr_of_trades":  nr_of_trades, "efficiency": efficiency, "optimal_efficiency": optimal_efficiency }
    loop_results.append(loop)

loop_results = pd.DataFrame(loop_results)







# Setting the display options
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)  # Increase the width to prevent wrapping

# Sorting and filtering
loop_results = loop_results.sort_values(by='profit_percentage', ascending=False)
top_50_results = loop_results.head(100)

# Printing the column names and data
print(symbol)
baseline_profit = ((technical_indicators["price"].iloc[-1]/technical_indicators["price"].iloc[1])-1)*100
print("BASELINE profit%", baseline_profit)
print(' '.join(top_50_results.columns))
print(top_50_results)






#Additional code that allowed to effectively visualise the results
plt.figure(figsize=(20,10))
plt.plot(technical_indicators['price'], label='Close Price', color='black', alpha=0.5)
plt.plot(technical_indicators['SMA_long'], label="SMA long", color='blue', alpha=0.35)
plt.plot(technical_indicators['SMA'], label="SMA", color='red', alpha=0.35)

# Plotting actual buy signals from the simulation
plt.scatter(technical_indicators.index, actual_flat_buy_signal, color='green', label='Flat Buy Signal', marker='o', alpha=1)
plt.scatter(technical_indicators.index, actual_hype_buy_signal, color='blue', label='Hype Buy Signal', marker='^', alpha=1)
plt.scatter(technical_indicators.index, actual_rsi_buy_signal, color='green', label='RSI Buy Signal', marker='s', alpha=1)

# Plotting actual sell signals from the simulation
plt.scatter(technical_indicators.index, actual_uptrend_sell_signal, color='red', label='Uptrend Sell Signal', marker='o', alpha=1)
plt.scatter(technical_indicators.index, actual_downtrend_sell_signal, color='red', label='Downtrend Sell Signal', marker='s', alpha=1)
plt.scatter(technical_indicators.index, actual_fall_sell_signal, color='red', label='Fall Sell Signal', marker='v', alpha=1)

# Shading based on SMA_long_direction
sma_direction = technical_indicators['SMA_long_direction']
plt.fill_between(technical_indicators.index, 0, technical_indicators['price'].max(), where=(sma_direction==1), interpolate=True, color='green', alpha=0.1, label='SMA rising')
plt.fill_between(technical_indicators.index, 0, technical_indicators['price'].max(), where=(sma_direction==0), interpolate=True, color='blue', alpha=0.1, label='SMA flat')
plt.fill_between(technical_indicators.index, 0, technical_indicators['price'].max(), where=(sma_direction==-1), interpolate=True, color='red', alpha=0.1, label='SMA falling')

plt.title('Stock price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# Second plot
plt.figure(figsize=(20, 3))
plt.plot(technical_indicators['TSI'], label='TSI', color='blue', alpha=0.5)
plt.plot(technical_indicators['TSI_signal'], label='TSI_signal', color='orange', alpha=0.5)

plt.title('TSI')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


