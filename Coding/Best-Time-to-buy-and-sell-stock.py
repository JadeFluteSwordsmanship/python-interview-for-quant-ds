'''
Say you have an array for which the ith element is the price of a given stock on day i. If you were only permitted to complete at most one transaction (one transaction including first buy one share of the stock in one day and then sell the one share of the stock in another day), design an algorithm to find the maximum profit with O(n) time complexity

Example 1:

Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5
Example 2:

Input: [7, 6, 4, 3, 1]
Output: 0
'''


def max_profit(price_list):
    buy_price = float('inf')
    max_profit = 0
    for i, price in enumerate(price_list):
        buy_price = min(buy_price, price)
        profit = price - buy_price
        max_profit = max(profit, max_profit)
    return max_profit

def max_profit(price_list):
    """
    Finds the maximum profit that can be made by buying and selling stock once.
    Time Complexity: O(n)
    :param price_list: List of stock prices
    :return: Maximum profit
    """
    if not price_list or len(price_list) < 2:
        return 0  # No profit if less than 2 prices exist

    min_price = price_list[0]  # Track the minimum price seen so far
    max_profit = 0  # Track the maximum profit

    for price in price_list[1:]:
        min_price = min(min_price, price)  # Update minimum price
        max_profit = max(max_profit, price - min_price)  # Update maximum profit

    return max_profit

print(max_profit([7, 1, 5, 3, 6, 4]))
