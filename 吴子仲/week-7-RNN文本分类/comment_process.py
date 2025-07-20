import csv
import jieba
import matplotlib.pyplot as plt
import pickle

db_comment = []
POSITIVE = 1
NEGATIVE = 0

with open('data/DMSC.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        words = jieba.lcut(row['Comment'])
        star = int(row['Star'])
        if star in [1, 2]:
            db_comment.append((words, NEGATIVE))
        elif star in [4, 5]:
            db_comment.append((words, POSITIVE))
        # 暂时只取前100000条数据
        if reader.line_num >= 100002:
            break

# 过滤掉长度大于10，小于100的评论
db_comment = [c for c in db_comment if len(c[0]) in range(10, 100)]

comment_len = [len(words) for words, _ in db_comment]

plt.hist(comment_len, bins=20, color='blue', alpha=0.7)
plt.xlabel('Comment Length')
plt.ylabel('Frequency')
plt.title('Comment Length Distribution')
plt.show()

with open('data/db_comment.pkl', 'wb') as f:
    pickle.dump(db_comment, f)
