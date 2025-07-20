import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
from comment_class_model import CommentsClassifier
from comments_corpus import build_collate_func, build_vocab_from_documents


# 数据集拆分
def train_test_split(X, y, split_rate=0.2):
    # 数据集拆分，默认比例0.2
    split_size = int(len(X) * (1 - split_rate))

    # 构建随机索引
    split_index = list(range(len(X)))
    # 打乱索引
    random.shuffle(split_index)

    # 拆分训练集和测试集
    x_train = [X[i] for i in split_index[:split_size]]
    y_train = [y[i] for i in split_index[:split_size]]
    x_test = [X[i] for i in split_index[split_size:]]
    y_test = [y[i] for i in split_index[split_size:]]

    return (x_train, y_train), (x_test, y_test)

# 模型训练
def train(model , train_dl , criterion , optimizer):
    global train_loss_cnt
    model.train()
    tpbar = tqdm(train_dl)
    for tokens , labels in tpbar:
        optimizer.zero_grad()
        tokens, labels = tokens.to(device), labels.to(device)
        loss = train_step(model, tokens, labels, criterion)
        loss.backward()
        optimizer.step()
        tpbar.set_description(f'Epoch: {epoch+1}, Train Loss: {loss.item():.4f}')
        # TensorBoard跟踪记录
        writer.add_scalar('train_loss', loss.item(), train_loss_cnt)
        train_loss_cnt += 1

# 计算损失
def train_step(model, tokens, labels, criterion):
    logits = model(tokens)
    loss = criterion(logits, labels)
    return loss

# 模型验证
def validate(model , val_dl , criterion):
    global val_loss_cnt , val_acc_cnt
    model.eval()
    tpbar = tqdm(val_dl)
    total_loss = 0
    total_acc = 0
    for tokens , labels in tpbar:
        tokens, labels = tokens.to(device), labels.to(device)
        loss, logits = validate_step(model, tokens, labels, criterion)
        # 用于在进度条中显示当前验证的损失和准确率
        tpbar.set_description(f'Epoch: {epoch+1}, Val Loss: {loss.item():.4f}, Val Acc: {total_acc/len(val_dl):.4f}')

        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == labels).float().mean()

        # TensorBoard跟踪记录
        # 损失值
        writer.add_scalar('val_avg_loss' , total_loss / len(val_dl), val_loss_cnt)
        val_loss_cnt += 1
        # 准确率
        writer.add_scalar('val_avg_acc' , total_acc / len(val_dl), val_acc_cnt)
        val_acc_cnt += 1

def validate_step(model , tokens , labels , criterion):
    with torch.no_grad():
        logits = model(tokens)
        loss = criterion(logits, labels)
    return loss, logits

if __name__ == '__main__':
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.cuda.is_available() else 'cpu'))

    # 定义tensorboard跟踪记录实现
    writer = SummaryWriter()
    # 跟踪相关变量
    train_loss_cnt = 0
    val_loss_cnt = 0
    val_acc_cnt = 0

    # 超参数设置
    BATCH_SIZE = 128  # 批次
    EPOCHS = 20  #  训练轮数
    EMBEDDING_SIZE = 200  # 词向量大小
    RNN_HIDDEN_SIZE = 200 # RNN隐藏层的维度
    LEARN_RATE = 1e-3  # 学习率
    NUM_LABELS = 2  # label数量 标签数量
    # 数据准备
    with open('./data/comments_jieba.bin', 'rb') as f:
        comments_jieba, labels = pickle.load(f)
    # 词典构建
    vocab = build_vocab_from_documents(comments_jieba)

    # 数据拆分
    (x_train, y_train), (x_test, y_test) = train_test_split(comments_jieba, labels)
    (x_valid, y_valid), (x_test, y_test) = train_test_split(x_test, y_test, split_rate=0.4)

    # 构建 Dataloader
    # 训练集
    train_dl = DataLoader(list(zip(x_train, y_train)), batch_size=BATCH_SIZE, collate_fn=build_collate_func(vocab), shuffle=True)
    # 验证集
    valid_dl = DataLoader(list(zip(x_valid, y_valid)), batch_size=BATCH_SIZE, collate_fn=build_collate_func(vocab), shuffle=True)
    # 测试集
    test_dl = DataLoader(list(zip(x_test, y_test)), batch_size=BATCH_SIZE, collate_fn=build_collate_func(vocab), shuffle=True)

    # 模型构建
    model = CommentsClassifier(
        vocab_size=len(vocab),
        emb_size=EMBEDDING_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        num_labels=NUM_LABELS
    )
    model.to(device)
    # 损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练数据
    for epoch in range(EPOCHS):
        train(model, train_dl, criterion, optimizer)
        validate(model, valid_dl, criterion)
    # 保存模型
    torch.save(model.state_dict(), 'comment_jieba_model.pth')
