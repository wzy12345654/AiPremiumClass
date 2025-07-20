# 打开修复后文件
fixed = open('book_comments_fix.txt', "w+", encoding='utf-8')

# 打开源文本文件
lines = [line for line in open('doubanbook_top250_comments.txt', "r", encoding='utf-8')]

for i, line in enumerate(lines):
    # 标题行直接写入
    if i == 0:
        fixed.write(line)
        pre_line = ''   #上一行置为空
        continue
    terms = line.split("\t")
    pre_terms = pre_line.split("\t")

    # 书名相同直接写入
    if terms[0] == pre_terms[0]:
        fixed.write(pre_line + "\n")   # 保存上一行
        pre_line = line.strip()     # 将当前行置为上一行
    # 书名不同时
    else:
        # 若为新书
        if len(terms) == 6:
            # 上一行数据异常则跳过
            if len(pre_terms) != 6:
                pre_line = line.strip()
                continue
            # 数据正常则写入
            fixed.write(pre_line + "\n")
            pre_line = line.strip()
        # 不为新书则将评论添加至上一行
        else:
            pre_line += line.strip()

# 最后一行需要手动处理
fixed.close()

