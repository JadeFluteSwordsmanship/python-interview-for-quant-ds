'''
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
Integers in each row are sorted from left to right in ascending order;
The first integer of each row is greater than the last integer of the previous row.



For example,
Consider the following matrix:
[
  [1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
Given target = 3, return true.
'''


def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False  # Edge case: empty matrix

    m, n = len(matrix), len(matrix[0])

    # Step 1: Binary Search on first column to find row
    top, bottom = 0, m - 1
    while top <= bottom:
        mid = (top + bottom) // 2
        if matrix[mid][0] == target:
            return True
        elif matrix[mid][0] < target:
            top = mid + 1
        else:
            bottom = mid - 1

    # `bottom` is now the last row where matrix[bottom][0] < target
    row = bottom
    if row < 0:
        return False  # If no valid row found

    # Step 2: Binary Search within the identified row
    left, right = 0, n - 1
    while left <= right:
        mid = (left + right) // 2
        if matrix[row][mid] == target:
            return True
        elif matrix[row][mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False  # Target not found


matrix = [
  [1,  3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]

print(search_matrix(matrix, 3))  # True
print(search_matrix(matrix, 13)) # False
print(search_matrix(matrix, 30)) # True
print(search_matrix(matrix, 1))  # True
print(search_matrix(matrix, 50)) # True
print(search_matrix(matrix, 100)) # False
