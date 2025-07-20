#修复后的文件
fixed = open("doubanbook_top250_comments_fixed.txt", "w", encoding="utf-8")

#修复前的文件
lines = [line for line in open("doubanbook_top250_comments.txt", "r", encoding="utf-8")]

for i,line in enumerate(lines):
    #保存标题列
    if i == 0:
        fixed.write(line)
        prev_line = ''
        continue
    #提取书名和文本
    terms = line.split("\t")
    #当前行的书名==上一行的书名
    if terms[0] == prev_line.split('\t')[0]:
        if len(prev_line.split('\t')) == 6:
        #保存上一行记录
            fixed.write(prev_line + '\n')
        #保存当前行记录
            prev_line = line.strip()
        else:
            prev_line = ''
    else:
        if len(terms) ==6:  #新书评论
            #保存当前行记录
            # fixed.write(line)
            prev_line = line.strip()
        else: 
            prev_line += line.strip()
fixed.close()
