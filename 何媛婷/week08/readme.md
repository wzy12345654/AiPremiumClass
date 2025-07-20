## 【第八周作业】

### 1. 使用中文对联数据集训练带有attention的seq2seq模型，利用tensorboard跟踪。
https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets
### 2. 尝试encoder hidden state不同的返回形式（concat和add）
### 3. 编写并实现seq2seq attention版的推理实现。

## 实现步骤
### 1. 环境准备
‌安装依赖‌：确保已安装Python及相关库
‌下载数据集‌：从Kaggle下载数据集，解压后应包含train和test文件夹。
### 2. 数据预处理
‌读取数据‌：从train文件夹中读取对联数据，通常数据格式为文本文件，每行一对对联（上联和下联）。
‌数据清洗‌：去除不必要的字符，如注释、特殊符号等，并进行分词处理（考虑使用jieba等中文分词工具）。
‌构建词汇表‌：统计所有词汇的频率，构建词汇表，并根据频率过滤低频词。
‌数据编码‌：将文本数据转换为序列索引，便于模型处理。
‌划分数据集‌：将数据集划分为训练集、验证集和测试集（如果test文件夹用作最终测试，则可从train中划分出验证集）。
### 3. 模型构建
‌Seq2Seq模型‌：使用TensorFlow的tf.keras.Sequential或tf.keras.Model子类化方法构建Seq2Seq模型。
‌Encoder‌：使用LSTM或GRU等循环神经网络作为编码器。
‌Decoder‌：同样使用LSTM或GRU，并加入Attention机制，以提高模型对输入序列不同部分的关注度。
‌Attention机制‌：实现自定义Attention层，或在TensorFlow的tf.keras.layers中寻找可用的Attention层。
‌损失函数与优化器‌：使用sparse_categorical_crossentropy作为损失函数，Adam优化器进行参数更新。
### 4. 模型训练
‌数据加载‌：使用tf.data.Dataset构建数据管道，实现批处理、打乱数据等功能。
‌回调函数‌：配置TensorBoard回调，以便在训练过程中记录日志。
‌训练模型‌：调用model.fit方法开始训练，传入训练数据、验证数据以及回调函数列表。
### 5. TensorBoard配置
‌启动TensorBoard‌：在训练前，确保TensorBoard已安装并可通过命令行启动。
‌日志目录‌：在模型训练代码中指定日志目录，TensorBoard将在此目录下记录训练过程中的各种指标。
‌查看TensorBoard‌：在浏览器中打开TensorBoard界面，通过指定日志目录查看训练过程中的损失、准确率等指标的变化趋势，以及模型图结构等信息。
### 6. 模型评估与测试
‌在验证集上评估‌：训练完成后，在验证集上评估模型性能，调整超参数以优化模型。
‌在测试集上测试‌：最终，在test文件夹中的测试集上测试模型，生成对联并评估生成质量。