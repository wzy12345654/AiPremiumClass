import fasttext

model = fasttext.train_supervised('comments.txt', epoch=10, dim=200)

print('文档词汇表长度', len(model.words))

# 获取词向量
print(model.get_word_vector('作者'))

# 获取近邻词
print(model.get_nearest_neighbors('作者'))

# 分析词间类比
print(model.get_analogies('作者', '故事', '真实'))

model2 = fasttext.train_supervised('cooking.stackexchange.txt', epoch=5, dim=100)

# 预测分类
# FastText.py报错，根据提示，尝试修改了np.array(obj, copy=False)为np.asarray(obj)，原因暂时不明
print(model2.predict("hello, food"))