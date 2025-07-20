import fasttext
import torch
from torch.utils.tensorboard import SummaryWriter

# 创建一个简单的无监督模型，注意不能使用中文目录
file_path = r"week5/gzr.txt"
#用于捕捉稀有词
#model = fasttext.train_unsupervised(file_path, model='skipgram')
#用于捕捉常见词
model = fasttext.train_unsupervised(file_path, model='cbow')
#print('分析"春秋婵" 近似词：', model.get_nearest_neighbors('春秋婵'))


#可视化
writer = SummaryWriter()
#是列表类型，内部是字符串
meta_data = model.words
# for item in meta_data:
#     print(meta_data)
#     break
embeddings = []
for word in meta_data:
    #将每个单词传化为词向量
    #embeddings = [
#     [0.1, -0.3, 0.5],   # "我"
#     [0.4, 0.2, -0.1],   # "喜欢"
#     [-0.2, 0.7, 0.3],   # "看"
#     [0.6, -0.4, 0.8],   # "电影"
#     ...
#       ]
    embeddings.append(model.get_word_vector(word))
# 
# metadata每个点的标签
writer.add_embedding(torch.tensor(embeddings), metadata=meta_data)

writer.close()