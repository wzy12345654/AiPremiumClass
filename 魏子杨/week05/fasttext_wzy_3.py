import fasttext

# 定义文件路径
file_path = r"week5/cooking.stackexchange.txt"

# 使用有监督学习方法训练模型
model = fasttext.train_supervised(
    input=file_path,
    #lr=0.5,               # 学习率
    epoch=25            # 迭代次数
   # wordNgrams=2,         # 使用2-gram特征
   # verbose=2            # 显示详细信息
)

print(model.predict("Which baking dish is best to bake a banana bread ?"))