'''
1.1You are climbing a stair case.
It takes n steps to reach to the top.
Each time you can either climb 1 or 2 steps.
Write a function with lowest time complexity to return how many distinct ways can you climb to the top?
'''

import time
import numpy as np

def climb_stair_On(n):
    """
    Computes the number of ways to climb `n` steps using O(n) dynamic programming.
    :param n: Number of steps
    :return: Number of distinct ways to reach the top
    """
    if n <= 2:
        return n  # f(1) = 1, f(2) = 2

    prev1, prev2 = 2, 1  # f(n-1), f(n-2)
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1

def climb_stair_Olgn(n):
    """
    Computes the number of ways to climb `n` steps using O(log n) matrix exponentiation.
    :param n: Number of steps
    :return: Number of distinct ways to reach the top
    """
    if n <= 2:
        return n

    # Fibonacci transformation matrix
    M = np.array([[1, 1], [1, 0]], dtype=object)

    def matrix_power(mat, exp):  # Fast exponentiation
        res = np.eye(len(mat), dtype=object)  # Identity matrix
        while exp:
            if exp % 2:
                res = np.dot(res, mat)
            mat = np.dot(mat, mat)
            exp //= 2
        return res

    return matrix_power(M, n)[0, 0]

# Test with time measurement
# n = 1_000_000  # Test value  # significantly faster using O(log n) fast exponentiation
n = 10  # Test value

start_time = time.time()
result_On = climb_stair_On(n)
time_On = time.time() - start_time

start_time = time.time()
result_Olgn = climb_stair_Olgn(n)
time_Olgn = time.time() - start_time

# Print results
print(f"climb_stair_On({n}) = {result_On}, Time: {time_On:.6f} seconds")
print(f"climb_stair_Olgn({n}) = {result_Olgn}, Time: {time_Olgn:.6f} seconds")

'''
Now, each time you can either climb 1 or 2,3…n steps. 
Write a function with lowest time complexity to return how many distinct ways can you climb to the top?
'''

import time

def climb_stair_crazy_On(n):
    """
    Computes the number of ways to climb `n` steps using O(n) dynamic programming.
    """
    if n == 0: return 1
    dp = [1] * (n + 1)  # dp[0] = 1, initialize all to 1
    for i in range(1, n + 1):
        dp[i] = sum(dp[:i])  # Recurrence relation: f(n) = f(n-1) + ... + f(0)
    return dp[n]

def climb_stair_crazy_O1(n):
    """
    Computes the number of ways to climb `n` steps when allowed to take 1 to `n` steps.
    Uses O(1) mathematical formula.
    """
    return 1 if n == 0 else 2 ** (n - 1)

# Set test value
n = 10

# Measure O(n) DP method time
start_time = time.time()
result_On = climb_stair_crazy_On(n)
time_On = time.time() - start_time

# Measure O(1) formula method time
start_time = time.time()
result_O1 = climb_stair_crazy_O1(n)
time_O1 = time.time() - start_time

# Print results
print(f"climb_stair_crazy_On({n}) = {result_On}, Time: {time_On:.6f} seconds")
print(f"climb_stair_crazy_O1({n}) = {result_O1}, Time: {time_O1:.6f} seconds")