## 作业内容
1. 使用豆瓣电影评论数据完成文本分类处理：文本预处理，加载、构建词典。
（评论得分1～2	表示positive取值：1，评论得分4～5代表negative取值：0）
https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments
2. 加载处理后文本构建词典、定义模型、训练、评估、测试。
3. 尝试不同分词工具进行文本分词，观察模型训练结果。

## 作业文件：
+ fetch_dicts.py 主要执行文件
+ fetch_LSTMModel.py 模型相关方法工具类
+ data_douban.py 数据加载相关方法工具类

## 作业结果返回：
```
Building prefix dict from the default dictionary ...
Loading model from cache C:\Users\HUAWEI\AppData\Local\Temp\jieba.cache
Loading model cost 0.592 seconds.
Prefix dict has been built successfully.
Epoch 1/5, Loss: 0.27104410946057156
Epoch 2/5, Loss: 0.22446187994540903
Epoch 3/5, Loss: 0.2119757397201851
Epoch 4/5, Loss: 0.20501224029344467
Epoch 5/5, Loss: 0.2004171389504728
Accuracy: 0.910036352620418
```
## 步骤整理：
1. 准备数据： 加载数据集、筛选数据、定义标签并将文本分词
2. 构建词典： 统计词频、词频排序、选词、添加特殊词、构建映射
3. 训练集数据准备：定义数据集、转换数据为张量
4. 定义模型：损失器和优化器
5. 训练模型：设置训练参数并对结果进行评估
