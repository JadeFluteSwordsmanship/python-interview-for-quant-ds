'''
题目：给定一个单词数组 words 和一个最大宽度 maxWidth，重新排版单词使其每行恰好有 maxWidth 个字符，且左右两端对齐（最后一行左对齐）。要求尽可能均匀分配单词间的空格，无法均匀分配时左侧空格数多于右侧。
示例：
输入：
words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
输出：
[
"This is an",
"example of text",
"justification. "
]
'''
def full_justify(words, maxWidth):
    def split_long_words(words):
        res = []
        for word in words:
            if len(word) <= maxWidth:
                res.append(word)
            else:
                for i in range(0, len(word), maxWidth):
                    res.append(word[i:i+maxWidth])
        return res

    def justify_line(line_words, maxWidth, is_last_line=False):
        if is_last_line or len(line_words) == 1:
            # 左对齐 + 末尾空格填充
            return ' '.join(line_words).ljust(maxWidth)

        total_words_len = sum(len(w) for w in line_words)
        spaces = maxWidth - total_words_len
        gaps = len(line_words) - 1
        space_base = spaces // gaps
        extra = spaces % gaps

        line = ""
        for i, word in enumerate(line_words):
            line += word
            if i < gaps:
                line += ' ' * (space_base + (1 if i < extra else 0))
        return line

    words = split_long_words(words)
    result = []
    current_line = []
    current_len = 0

    for word in words:
        if current_len + len(current_line) + len(word) > maxWidth:
            # 已有的词 + 空格 + 新词会超
            result.append(justify_line(current_line, maxWidth))
            current_line = [word]
            current_len = len(word)
        else:
            current_line.append(word)
            current_len += len(word)

    # 最后一行
    result.append(justify_line(current_line, maxWidth, is_last_line=True))
    return result

words = ["This", "is", "an", "example", "of", "text", "justification."]
maxWidth = 16
output = full_justify(words, maxWidth)
for line in output:
    print(f'"{line}"')