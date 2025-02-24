'''
  This code exercise is to simulate exchange for matching orders.
  - When an order is input, an unique order Id is assigned to the order.
  - Orders are matched in the ordering of price and time priorities.
  Please finish class Exchange to fulfil the task in main function.
'''

import heapq
import itertools


class Exchange:
    def __init__(self):
        """
        Initializes the exchange with:
        - buy_orders: max-heap for buy orders (sorted by -price for max-heap behavior).
        - sell_orders: min-heap for sell orders (sorted by price for min-heap behavior).
        - order_trades: dictionary to track order execution details.
        - order_id_counter: unique order ID generator.
        """
        self.buy_orders = []  # Max-Heap (negative price for highest priority)
        self.sell_orders = []  # Min-Heap (sorted by price)
        self.order_trades = {}  # Stores {orderId: (tradeVolume, avgPrice)}
        self.order_id_counter = itertools.count(1)  # Auto-increment order ID

    def InputOrder(self, side, volume, price):
        """
        Receives an order and attempts to match it. If no match, adds to the order book.
        :param side: 0 for buy, 1 for sell (-1 is invalid)
        :param volume: Order quantity
        :param price: Order price
        :return: Order ID
        """
        if side not in [0, 1] or volume <= 0 or price <= 0:
            return -1  # Invalid order

        order_id = next(self.order_id_counter)  # Generate a unique order ID
        remaining_volume = volume
        total_cost = 0  # Total cost of executed trades

        # Buy Order Processing (Try matching with sell orders)
        if side == 0:
            while self.sell_orders and remaining_volume > 0:
                sell_price, sell_time, sell_id, sell_volume = self.sell_orders[0]
                if sell_price > price:  # No more matching possible
                    break

                heapq.heappop(self.sell_orders)  # Remove matched sell order
                matched_volume = min(remaining_volume, sell_volume)
                total_cost += matched_volume * sell_price  # Compute trade value
                remaining_volume -= matched_volume
                sell_volume -= matched_volume

                # Update trade records
                self.order_trades.setdefault(order_id, [0, 0])
                self.order_trades[order_id][0] += matched_volume  # Update trade volume
                self.order_trades[order_id][1] += matched_volume * sell_price  # Update trade value

                self.order_trades.setdefault(sell_id, [0, 0])
                self.order_trades[sell_id][0] += matched_volume
                self.order_trades[sell_id][1] += matched_volume * sell_price

                if sell_volume > 0:
                    heapq.heappush(self.sell_orders, (sell_price, sell_time, sell_id, sell_volume))

        # Sell Order Processing (Try matching with buy orders)
        else:
            while self.buy_orders and remaining_volume > 0:
                buy_price, buy_time, buy_id, buy_volume = self.buy_orders[0]
                if buy_price < price:  # No more matching possible
                    break

                heapq.heappop(self.buy_orders)  # Remove matched buy order
                matched_volume = min(remaining_volume, buy_volume)
                total_cost += matched_volume * buy_price
                remaining_volume -= matched_volume
                buy_volume -= matched_volume

                # Update trade records
                self.order_trades.setdefault(order_id, [0, 0])
                self.order_trades[order_id][0] += matched_volume  # Update trade volume
                self.order_trades[order_id][1] += matched_volume * buy_price  # Update trade value

                self.order_trades.setdefault(buy_id, [0, 0])
                self.order_trades[buy_id][0] += matched_volume
                self.order_trades[buy_id][1] += matched_volume * buy_price

                if buy_volume > 0:
                    heapq.heappush(self.buy_orders, (-buy_price, buy_time, buy_id, buy_volume))

        # If remaining volume exists, add to order book
        if remaining_volume > 0:
            if side == 0:
                heapq.heappush(self.buy_orders, (-price, order_id, order_id, remaining_volume))
            else:
                heapq.heappush(self.sell_orders, (price, order_id, order_id, remaining_volume))

        return order_id

    def QueryOrderTrade(self, orderId):
        """
        Queries an order's trade volume and average price.
        :param orderId: Assigned order ID
        :return: Tuple (trade volume, average price)
        """
        trade_volume, total_cost = self.order_trades.get(orderId, (0, 0))
        avg_price = total_cost / trade_volume if trade_volume > 0 else 0
        return trade_volume, avg_price


if __name__ == "__main__":
    ex = Exchange()
    orders = []

    # Input Orders
    orders.append(ex.InputOrder(0, 1, 100))  # Buy 1 @ 100
    orders.append(ex.InputOrder(0, 2, 101))  # Buy 2 @ 101
    orders.append(ex.InputOrder(0, 3, 102))  # Buy 3 @ 102
    orders.append(ex.InputOrder(1, 4, 100))  # Sell 4 @ 100 (Matches buy 1 @ 100 + buy 2 @ 101 + part of buy 3 @ 102)
    orders.append(ex.InputOrder(1, 5, 101))  # Sell 5 @ 101 (Matches remaining buy 3 @ 102)
    orders.append(ex.InputOrder(1, 6, 102))  # Sell 6 @ 102 (New order in sell book)

    # Query Trade Results
    for orderId in orders:
        tradeVolume, avgPrice = ex.QueryOrderTrade(orderId)
        print(f"orderId:{orderId}, tradeVolume:{tradeVolume}, averagePrice:{avgPrice:.2f}")
