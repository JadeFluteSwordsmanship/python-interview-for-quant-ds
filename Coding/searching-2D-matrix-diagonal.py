'''
二维数组中的查找
在一个二维数组array中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
[
[1,2,8,9],
[2,4,9,12],
[4,7,10,13],
[6,8,11,15]
]
给定 target = 7，返回 true。

给定 target = 3，返回 false。

数据范围：矩阵的长宽满足0<=n,m<=500， 矩阵中的值满足 0<=val<=10**9
进阶：空间复杂度O(1)，时间复杂度O(n+m)
'''

def find_number_in_2d_array(matrix, target):
    if not matrix or not matrix[0]:
        return False

    rows = len(matrix)
    cols = len(matrix[0])
    row = 0
    col = cols - 1  # 从右上角开始

    while row < rows and col >= 0:
        value = matrix[row][col]
        if value == target:
            return True
        elif value > target:
            col -= 1  # 左移
        else:
            row += 1  # 下移
    return False

