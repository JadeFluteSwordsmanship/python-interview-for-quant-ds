'''
Given a string of parentheses (e.g., "(()()(("), remove the minimum number of characters to make the parentheses valid (i.e., properly matched).

The program should find all possible longest valid substrings that can be obtained by removing the minimum number of characters.

Example:
Input: '()(()))))'
Output: ()(()), ((()))
(We need to remove three ) characters)
'''


def remove_invalid_parentheses(s):
    def get_min_remove(s):
        """ 计算最少需要删除多少个字符才能使括号匹配 """
        left, right = 0, 0
        for char in s:
            if char == '(':
                left += 1
            elif char == ')':
                if left > 0:
                    left -= 1  # 配对一个 '('
                else:
                    right += 1  # 额外的 ')'
        return left, right  # 需要删除的 '(' 和 ')' 数量

    def dfs(index, left_count, right_count, left_rem, right_rem, expr):
        """ 回溯找出所有可能的合法字符串 """
        if index == len(s):
            if left_rem == 0 and right_rem == 0:
                valid_result.add("".join(expr))  # 记录合法结果
            return

        char = s[index]

        # **尝试删除当前字符**
        if char == '(' and left_rem > 0:
            dfs(index + 1, left_count, right_count, left_rem - 1, right_rem, expr)
        if char == ')' and right_rem > 0:
            dfs(index + 1, left_count, right_count, left_rem, right_rem - 1, expr)

        # **尝试保留当前字符**
        expr.append(char)
        if char == '(':
            dfs(index + 1, left_count + 1, right_count, left_rem, right_rem, expr)
        elif char == ')':
            if left_count > right_count:  # 只有当前 ')' 能配对时才加入
                dfs(index + 1, left_count, right_count + 1, left_rem, right_rem, expr)
        else:
            dfs(index + 1, left_count, right_count, left_rem, right_rem, expr)

        # **回溯，恢复状态**
        expr.pop()

    left_remove, right_remove = get_min_remove(s)  # 计算最少删除字符
    valid_result = set()
    dfs(0, 0, 0, left_remove, right_remove, [])

    return list(valid_result), left_remove + right_remove


# 测试
if __name__ == "__main__":
    test_str = "()(()))))"
    result, min_remove = remove_invalid_parentheses(test_str)
    print(f"最少需要移除 {min_remove} 个括号。")
    print("所有可能的最长匹配子串:")
    for r in result:
        print(r)
