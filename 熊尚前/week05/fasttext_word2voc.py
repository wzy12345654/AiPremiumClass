import fasttext
import jieba
import torch
from torch.utils.tensorboard import SummaryWriter

# tokennizer = lambda x: jieba.lcut(x)

# with open('流浪地球.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# words = tokennizer(text)
# with open('liulangdiqiu.txt', 'w', encoding='utf-8') as f:
#     for word in words:
#         f.write(word + ' ')


# 使用绝对路径指定文件
file_path = 'liulangdiqiu.txt'

model = fasttext.train_unsupervised(file_path, epoch=100, lr=0.1, dim=300)
print('“逻辑”的近似词是：',model.get_nearest_neighbors('逻辑')[0][1])
print('“太阳”的近似词是：',model.get_nearest_neighbors('太阳')[0][1])
print('“二向箔”的近似词是：',model.get_nearest_neighbors('二向箔')[0][1])
print('“贾宝玉”的近似词是：',model.get_nearest_neighbors('贾宝玉'))


writer = SummaryWriter()
meta_data = model.words  # 元数据
embeddings = []
for word in meta_data:
    embeddings.append(model.get_word_vector(word))  # 词向量
writer.add_embedding(torch.tensor(embeddings), metadata=meta_data)  # 添加词向量和元数据
writer.close()  # 关闭 SummaryWriter 对象，确保数据写入文件