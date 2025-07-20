import sentencepiece as spm

# 引入外部文本资料训练分词模型
# spm.SentencePieceTrainer.Train(
#     input='data/DMSC.csv',  # 输入文件路径
#     model_prefix='data/DMSC_mod',  # 输出模型文件前缀
#     vocab_size=8000)  # 词汇表大小

# 加载模型进行分词
sp = spm.SentencePieceProcessor(model_file='data/DMSC_mod.model')
# 测试分词
print(sp.EncodeAsPieces('这部电影有点像疯狂动物城'))  # 输出分词结果