import jieba
import fasttext

# #文档分词预处理
# with open('XYJ.txt' , 'r', encoding='utf-8') as f:
#     #.redlines()方法可以读取文本中的所有行
#     lines = f.read()

# with open('XYJ_cut.txt', 'w', encoding='utf-8') as f:
#     f.write(' '.join(jieba.lcut(lines)))

#模型构建
model = fasttext.train_unsupervised('XYJ_cut.txt', model='skipgram')


#获取词向量
print(model.get_word_vector('孙悟空'))

##获取词向量的最近邻
print(model.get_nearest_neighbors('孙悟空', k=5))

#分析词间类比
print(model.get_analogies('孙悟空', '唐僧', '猪八戒', k=5))

# #模型保存
# model.save_model('XYJ_skipgram.bin')

# #模型加载
# model = fasttext.load_model('XYJ_skipgram.bin')