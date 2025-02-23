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

    position = 0  # 持仓量
    avg_price = 0  # 持仓的平均成本价
    realized_pnl = 0  # 已实现盈亏
    pnl_snapshots = []  # 存储 PnL 变化快照

    trade_index = 0  # 交易数据索引
    trade_count = len(trade_df)

    for i, row in price_df.iterrows():
        timestamp = row["timestamp"]
        last_price = row["lastPrice"]

        # 处理当前时间点前发生的交易
        while trade_index < trade_count and trade_df.loc[trade_index, "tradeTime"] <= timestamp:
            trade = trade_df.loc[trade_index]
            side, volume, trade_price = trade["side"], trade["volume"], trade["price"]

            if side == "B":  # 买入
                total_cost = avg_price * position + trade_price * volume
                position += volume
                avg_price = total_cost / position if position > 0 else 0
            elif side == "S":  # 卖出
                if position >= volume:
                    realized_pnl += (trade_price - avg_price) * volume
                    position -= volume
                else:
                    print(f"Warning: Selling more than holding at {timestamp}")

            trade_index += 1  # 处理下一个交易

        # 计算当前时间的浮动盈亏
        unrealized_pnl = (last_price - avg_price) * position
        total_pnl = realized_pnl + unrealized_pnl

        pnl_snapshots.append([timestamp, total_pnl])

    pnl_df = pd.DataFrame(pnl_snapshots, columns=["timestamp", "PnL"])

    plt.figure(figsize=(10, 5))
    plt.plot(pnl_df["timestamp"], pnl_df["PnL"], label="PnL Over Time", color="blue")
    plt.xlabel("Timestamp")
    plt.ylabel("PnL")
    plt.title("PnL Curve Over Time")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)

    plt.savefig(output, bbox_inches="tight")
    plt.close()

    print(f"PnL curve saved to {output}")


# 执行主程序
if __name__ == "__main__":
    calc_pnl_and_draw_picture("./data/price.csv", "./data/trade.csv", "./data/pnl.png")
