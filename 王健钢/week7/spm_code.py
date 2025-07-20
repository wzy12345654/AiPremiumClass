import sentencepiece as spm

# 引入外部文本资料训练分词模型
# spm.SentencePieceTrainer.Train(input='hlm_c.txt', 
#                                model_prefix='hlm_mod',
#                                vocab_size=10000) # Train the model

# 加载模型进行分词
sp = spm.SentencePieceProcessor(model_file='hlm_mod.model')
# 测试分词
print(sp.EncodeAsPieces('尤氏的母亲并邢夫人、王夫人、凤姐儿都吃毕饭，漱了口，净了手，才说要往园子里去。')) # 分词