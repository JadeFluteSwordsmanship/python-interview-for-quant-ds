import pandas as pd
import matplotlib.pyplot as plt


def calc_pnl_and_draw_picture(price_file, trade_file, output):
    """
    Read price and trade data, calculate PnL over time, and save a PnL curve.

    :param price_file: CSV file containing price data
    :param trade_file: CSV file containing trade data
    :param output: Path to save the PnL curve plot
    :return: None
    """

    price_df = pd.read_csv(price_file, parse_dates=["timestamp"])
    trade_df = pd.read_csv(trade_file, parse_dates=["tradeTime"])

    position = 0
    avg_price = 0
    realized_pnl = 0
    pnl_snapshots = []

    trade_index = 0
    trade_count = len(trade_df)

    for i, row in price_df.iterrows():
        timestamp = row["timestamp"]
        last_price = row["lastPrice"]

        while trade_index < trade_count and trade_df.loc[trade_index, "tradeTime"] <= timestamp:
            trade = trade_df.loc[trade_index]
            side, volume, trade_price = trade["side"], trade["volume"], trade["price"]

            if side == "B":  
                if position < 0:  
                    if abs(position) >= volume:  
                        realized_pnl += (avg_price - trade_price) * volume  
                        position += volume
                    else:  
                        realized_pnl += (avg_price - trade_price) * abs(position)  
                        long_volume = volume - abs(position)  
                        position = long_volume
                        avg_price = trade_price
                else:  
                    total_cost = avg_price * position + trade_price * volume
                    position += volume
                    avg_price = total_cost / abs(position) if position != 0 else 0

            elif side == "S":  
                if position > 0:  
                    if position >= volume:  
                        realized_pnl += (trade_price - avg_price) * volume
                        position -= volume
                    else:  
                        realized_pnl += (trade_price - avg_price) * position  
                        short_volume = volume - position  
                        position = -short_volume
                        avg_price = trade_price  
                else:  
                    total_cost = avg_price * position + trade_price * volume
                    position -= volume  
                    avg_price = total_cost / abs(position) if position != 0 else 0

            trade_index += 1

        unrealized_pnl = (last_price - avg_price) * position
        total_pnl = realized_pnl + unrealized_pnl

        pnl_snapshots.append([timestamp, total_pnl])

    pnl_df = pd.DataFrame(pnl_snapshots, columns=["timestamp", "PnL"])

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df["timestamp"], pnl_df["PnL"], label="PnL", color="blue")
    plt.xlabel("Timestamp")
    plt.ylabel("PnL")
    plt.title("PnL Curve Over Time")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    plt.savefig(output, bbox_inches="tight")
    plt.close()

    print(f"PnL curve saved to {output}")
    return pnl_snapshots

if __name__ == "__main__":
    curve = calc_pnl_and_draw_picture("./data/price.csv", "./data/trade.csv", "./data/pnl.png")
    # print(curve)