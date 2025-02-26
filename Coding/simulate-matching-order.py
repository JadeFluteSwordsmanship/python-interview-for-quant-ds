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
        self.sellHeap = []
        self.buyHeap = []
        # heap saves time O(logN) so we don't sort by price and time for every match which costs O(NlogN)
        self.orders = {}
        self.order_id_counter = itertools.count(1)

    def InputOrder(self, side, volume, price):
        """
        InputOrder receives order, and return assigned order Id.
        :param side: 0 is buy, 1 is sell,-1 means not valid
        :param volume:order's quantity
        :param price:order's prices
        :return: order Id, an integer
        """
        if side not in [0, 1] or volume <= 0 or price <= 0:
            return -1  # Invalid order

        orderId = next(self.order_id_counter)
        self.orders[orderId] = {
            'side': side,
            'price': price,
            'initialVolume': volume,
            'remainingVolume': volume,
            'filledVolume': 0,
            'filledCost': 0.0
        }
        if side == 0:
            self.matchBuy(orderId)
            rem = self.orders[orderId]['remainingVolume']
            if rem > 0:
                heapq.heappush(self.buyHeap, (-price, orderId, rem))
        else:
            self.matchSell(orderId)
            rem = self.orders[orderId]['remainingVolume']
            if rem > 0:
                heapq.heappush(self.sellHeap, (price, orderId, rem))

        return orderId

    def matchBuy(self, buyOrderId):
        buyPrice = self.orders[buyOrderId]['price']
        buyRem = self.orders[buyOrderId]['remainingVolume']

        while buyRem > 0 and self.sellHeap:
            bestSellPrice, bestSellId, bestSellVol = self.sellHeap[0]

            if bestSellPrice > buyPrice:
                break

            tradedVol = min(buyRem, bestSellVol)
            tradePrice = bestSellPrice  # 主动买 以挂单价成交

            self.orders[buyOrderId]['filledVolume'] += tradedVol
            self.orders[buyOrderId]['filledCost'] += tradedVol * tradePrice
            self.orders[buyOrderId]['remainingVolume'] -= tradedVol
            buyRem -= tradedVol

            self.orders[bestSellId]['filledVolume'] += tradedVol
            self.orders[bestSellId]['filledCost'] += tradedVol * tradePrice
            self.orders[bestSellId]['remainingVolume'] -= tradedVol

            if self.orders[bestSellId]['remainingVolume'] == 0:
                heapq.heappop(self.sellHeap)
            else:
                newVol = self.orders[bestSellId]['remainingVolume']
                heapq.heapreplace(self.sellHeap, (bestSellPrice, bestSellId, newVol))

    def matchSell(self, sellOrderId):
        sellPrice = self.orders[sellOrderId]['price']
        sellRem = self.orders[sellOrderId]['remainingVolume']

        while sellRem > 0 and self.buyHeap:
            negBuyPrice, bestBuyId, bestBuyVol = self.buyHeap[0]
            bestBuyPrice = -negBuyPrice

            if bestBuyPrice < sellPrice:
                break

            tradedVol = min(sellRem, bestBuyVol)
            tradePrice = bestBuyPrice # 被动买，以买单挂单价成交

            self.orders[sellOrderId]['filledVolume'] += tradedVol
            self.orders[sellOrderId]['filledCost'] += tradedVol * tradePrice
            self.orders[sellOrderId]['remainingVolume'] -= tradedVol
            sellRem -= tradedVol

            self.orders[bestBuyId]['filledVolume'] += tradedVol
            self.orders[bestBuyId]['filledCost'] += tradedVol * tradePrice
            self.orders[bestBuyId]['remainingVolume'] -= tradedVol

            if self.orders[bestBuyId]['remainingVolume'] == 0:
                heapq.heappop(self.buyHeap)
            else:
                newVol = self.orders[bestBuyId]['remainingVolume']
                heapq.heapreplace(self.buyHeap, (negBuyPrice, bestBuyId, newVol))
    def QueryOrderTrade(self, orderId):
        """
        Queries an order's trade volume and average price.
        :param orderId: Assigned order ID
        :return: Tuple (trade volume, average price)
        """
        if orderId not in self.orders:
            return (0, 0.0)

        filledVol = self.orders[orderId]['filledVolume']
        filledCost = self.orders[orderId]['filledCost']
        if filledVol > 0:
            avgPrice = filledCost / filledVol
        else:
            avgPrice = 0.0
        return (filledVol, avgPrice)


if __name__ == "__main__":
    ex = Exchange()
    orders = list()

    orders.append(ex.InputOrder(0, 1, 100))
    orders.append(ex.InputOrder(0, 2, 101))
    orders.append(ex.InputOrder(0, 3, 102))
    orders.append(ex.InputOrder(1, 4, 100))
    orders.append(ex.InputOrder(1, 5, 101))
    orders.append(ex.InputOrder(1, 6, 102))

    for orderId in orders:
        tradeVolume, avgPrice = ex.QueryOrderTrade(orderId)
        print("orderId:%d,tradeVolume:%d,averagePrcies:%f"%(orderId,tradeVolume,avgPrice))
