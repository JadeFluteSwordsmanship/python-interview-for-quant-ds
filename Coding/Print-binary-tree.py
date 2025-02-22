'''
1.Print binary tree

implement level print function, which print binary tree level by level
and print even level from right to left while odd level from left to right

example:
supposed there is a binary tree like below
              a          ---- level 0
            /   \
           b     c       ---- level 1
          / \   / \
         d   e f   g     ---- level 2
        / \
       h   i             ---- level 3

output should be this: abcgfedh

TreeNode class is the binary tree node
'''
from collections import deque


class TreeNode:
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right


def level_print(tree_root):
    if not tree_root:
        return

    queue = deque([tree_root])  # 初始化队列
    level = 0  # 层计数器
    result = []  # 存储最终输出

    while queue:
        level_size = len(queue)  # 当前层的节点个数
        current_level = deque()  # 用 deque 来存储当前层的节点数据

        for _ in range(level_size):
            node = queue.popleft()  # 取出当前层的节点

            # 按层次存储
            if level % 2 == 1:
                current_level.append(node.data)  # 奇数层（从左往右存）
            else:
                current_level.appendleft(node.data)  # 偶数层（从右往左存）

            # 加入子节点到队列（始终从左到右加入）
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # 记录当前层的结果
        result.extend(current_level)
        level += 1  # 进入下一层

    print("".join(result))  # 按要求输出

if __name__ == "__main__":
    root = TreeNode('a')
    root.left = TreeNode('b')
    root.right = TreeNode('c')
    root.left.left = TreeNode('d')
    root.left.right = TreeNode('e')
    root.right.left = TreeNode('f')
    root.right.right = TreeNode('g')
    root.left.left.left = TreeNode('h')
    root.left.left.right = TreeNode('i')

    print("预期输出: abcgfedhi")
    print("实际输出: ", end="")
    level_print(root)