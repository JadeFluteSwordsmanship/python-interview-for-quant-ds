
'''
随着游戏的不断发展，技能系统变得越来越复杂化，不同的技能分支可以形成千变万化的技能搭配。
作为一名“游戏热爱者”，你决定开发一款名为“技能加点大师”的小工具，快速地计算出将所有可用技能点全部消耗的情况下，一共可以有多少种可行的技能加点方式。
技能系统规则：
每个技能每次加点均需要消耗一定数量的技能点。
当可用技能点少于某个技能单次加点所需消耗的技能点时，不可对该技能加点。
每个技能被加点的次数没有上限。

输入描述：
第 1 行包含两个整数N和M ，分别表示技能个数和玩家当前拥有的技能点数。
第 2 行包含P个整数 ，表示每个技能每次加点需要消耗的技能点数。

输出描述：
输出一个整数，表示玩家正好将所有技能点完全消耗时，可以有多少种技能加点方式。
'''
import sys


def count_skill_point_ways():
    # 读取N和M
    N, M = map(int, sys.stdin.readline().strip().split())

    # 读取每个技能的技能点消耗
    P = list(map(int, sys.stdin.readline().strip().split()))

    # 使用动态规划计算组合方式
    dp = [0] * (M + 1)
    dp[0] = 1

    # 遍历每个技能消耗
    for p in P:
        for i in range(p, M + 1):
            dp[i] += dp[i - p]

    print(dp[M])


if __name__ == "__main__":
    count_skill_point_ways()
