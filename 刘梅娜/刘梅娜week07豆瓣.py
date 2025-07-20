import pandas as pd
import re
import jieba
import thulac

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 加载数据
data = pd.read_csv('DMSC.csv')
# 去除空值
data.dropna(inplace=True)
# 去除重复值
data.drop_duplicates(inplace=True)

# 标注数据
data['sentiment'] = data['Star'].apply(lambda x: 1 if x in [1, 2] else  0)

# 去除标点符号和转换为小写
data['Comment'] = data['Comment'].apply(lambda x: re.sub(r'[^\w\s]', '', x).lower())

# 使用jieba进行分词
data['Comment'] = data['Comment'].apply(lambda x: ' '.join(jieba.lcut(x)))

# 构建词典
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Comment'])
y = data['sentiment']


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
model = MultinomialNB()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))



# 初始化thulac
thu1 = thulac.thulac()

# 使用thulac进行分词
data['comment_thulac'] = data['Comment'].apply(lambda x: ' '.join([word for word, tag in thu1.cut(x)]))

# 构建词典
vectorizer_thulac = CountVectorizer()
X_thulac = vectorizer_thulac.fit_transform(data['comment_thulac'])
y_thulac = data['sentiment']

# 划分训练集和测试集
X_train_thulac, X_test_thulac, y_train_thulac, y_test_thulac = train_test_split(X_thulac, y_thulac, test_size=0.2, random_state=42)

# 定义模型
model_thulac = MultinomialNB()

# 训练模型
model_thulac.fit(X_train_thulac, y_train_thulac)

# 预测
y_pred_thulac = model_thulac.predict(X_test_thulac)

# 评估模型
print("Accuracy with Thulac:", accuracy_score(y_test_thulac, y_pred_thulac))
print("Classification Report with Thulac:\n", classification_report(y_test_thulac, y_pred_thulac))




