"""
given symbol_list,like ["A","B"，"C"]
given datetime_start,like "20180103 09:00:00"
given datetime_end,like "20180103 15:00:00"
given column_list,like ["last_price", "ask1", "bid1"]

generate random price in column list for every symbol
from datetime start to datetime end sample every second

       last_price      ask1      bid1 symbol                time
0        0.173368  0.666229  0.650586      A 2018-01-03 09:00:00
1        0.434612  0.771833  0.891474      A 2018-01-03 09:00:01
2        0.222062  0.891160  0.299797      A 2018-01-03 09:00:02
3        0.947866  0.073749  0.169601      A 2018-01-03 09:00:03
4        0.800996  0.548314  0.360005      A 2018-01-03 09:00:04
5        0.529903  0.542695  0.876590      A 2018-01-03 09:00:05

"""
import pandas as pd
import numpy as np


def generate_data(symbol_list, datetime_start, datetime_end, column_list):
    """
    Generate random price data every second for each symbol.

    :param symbol_list: List of symbols (e.g., ["A", "B", "C"])
    :param datetime_start: Start datetime string (e.g., "20180103 09:00:00")
    :param datetime_end: End datetime string (e.g., "20180103 15:00:00")
    :param column_list: Columns to generate prices (e.g., ["last_price", "ask1", "bid1"])
    :return: DataFrame with simulated price data
    """
    time_range = pd.date_range(start=pd.to_datetime(datetime_start, format="%Y%m%d %H:%M:%S"),
                               end=pd.to_datetime(datetime_end, format="%Y%m%d %H:%M:%S"),
                               freq='S')  # Every second

    df_list = []
    for symbol in symbol_list:
        df = pd.DataFrame(index=time_range)
        df["symbol"] = symbol
        df["time"] = df.index
        for col in column_list:
            df[col] = np.random.rand(len(df))  # Generate random prices
        df_list.append(df)

    return pd.concat(df_list).reset_index(drop=True)

"""
resample the last price column in second to generate open high low close in minutes
generate bar return (bar_ret) column which means close /open
generate flag column: if high / close <1.5 flag=1 else flag = 0
no for loop statement 
                          open      high       low     close    bar_ret  flag
symbol time                                                                        
A 2018-01-03 09:01:00  0.256674  0.996918  0.010180  0.082958   0.323205     0
  2018-01-03 09:02:00  0.114607  0.984437  0.033200  0.409413   3.572324     0
  2018-01-03 09:03:00  0.261541  0.980200  0.002990  0.462317   1.767662     0
  2018-01-03 09:04:00  0.379665  0.949153  0.006085  0.594945   1.567028     0
  2018-01-03 09:05:00  0.029332  0.987624  0.001219  0.314411  10.718877     0
"""
def cal_min_bar(df):
    """
    Resample last_price column per minute to generate open, high, low, close (OHLC).
    Compute bar_ret (close/open) and flag (1 if high/close < 1.5, else 0).

    :param df: DataFrame containing ["time", "symbol", "last_price"]
    :return: DataFrame with OHLC, bar_ret, and flag
    """
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    # Resample to minute bars
    bar_data = df.groupby("symbol")["last_price"].resample("1T").agg(["first", "max", "min", "last"])

    # Rename columns
    bar_data.columns = ["open", "high", "low", "close"]

    # Compute bar_ret and flag
    bar_data["bar_ret"] = bar_data["close"] / bar_data["open"]
    bar_data["flag"] = (bar_data["high"] / bar_data["close"] < 1.5).astype(int)

    return bar_data.reset_index()


def select_data(df, n):
    """
    Select the top N rows with the highest bar_ret where flag == 1 for each symbol.

    :param df: DataFrame containing ["symbol", "time", "bar_ret", "flag"]
    :param n: Number of top rows to select per symbol
    :return: Filtered DataFrame
    """
    return (df[df["flag"] == 1]
    .groupby("symbol")
    .apply(lambda x: x.nlargest(n, "bar_ret"))
    .reset_index(drop=True)
    [["symbol", "time", "bar_ret"]])


if __name__ == "__main__":
    symbol_list = ["A", "B", "C"]
    start = "20180103 09:00:00"
    end = "20180103 15:00:00"
    column_list = ["last_price", "ask1", "bid1"]

    price_data = generate_data(symbol_list, start, end, column_list)
    print(price_data.head())

    bar_data = cal_min_bar(price_data[["time", "symbol", "last_price"]])
    print(bar_data.head())

    focus_data = select_data(bar_data, 3)
    print(focus_data)
