import fasttext

model = fasttext.train_supervised("cooking.stackexchange.txt")


#文本分类共功能
print(model.predict("bread What's the purpose of a bread box?"))