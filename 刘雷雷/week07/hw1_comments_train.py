import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import os
import random
from torch.utils.tensorboard import SummaryWriter
from hw1_comment_classifiler_model import CommentsClassifier
from hw1_comment_corpus_proc import build_collate_func, build_vocab_from_documents

current_dir = os.path.dirname(os.path.abspath(__file__))


def train_test_split(x, y, split_rate=0.2):
    # 数据拆分流程
    # 1.拆分比率
    # 2.样本随机性
    # 3.构建拆分索引
    # 4.借助slice拆分

    split_size = int(len(x) * (1 - split_rate))
    split_index = list(range(len(x)))
    random.shuffle(split_index)
    x_train = [x[i] for i in split_index[:split_size]]
    y_train = [y[i] for i in split_index[:split_size]]

    x_test = [x[i] for i in split_index[split_size:]]
    y_test = [y[i] for i in split_index[split_size:]]

    return (x_train, y_train), (x_test, y_test)


def train(model, train_dl, criterion, optimizer):
    global train_loss_cnt
    model.train()  # 设置模型为训练模式
    tpbar = tqdm(train_dl)
    for tokens, labels in tpbar:
        optimizer.zero_grad()  # 清除梯度

        tokens, labels = tokens.to(device), labels.to(device)

        loss = train_step(model, tokens, labels, criterion)
        loss.backward()
        optimizer.step()

        # 更新进度条
        tpbar.set_description(f"epoch: {epoch+1} Train Loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), train_loss_cnt)
        train_loss_cnt += 1


def train_step(model, tokens, labels, criterion):
    logits = model(tokens)  # 前向传播
    loss = criterion(logits, labels)  # 计算损失
    return loss  # 返回损失值


def validate(model, val_dl, criterion):
    global val_loss_cnt, val_acc_cnt
    model.eval()  # 设置模型为评估模式
    tpbar = tqdm(val_dl)
    total_loss = 0
    total_acc = 0

    for tokens, labels in tpbar:
        tokens, labels = tokens.to(device), labels.to(device)

        loss, logits = validate_step(model, tokens, labels, criterion)
        tpbar.set_description(f"epoch: {epoch+1} Val Loss: {loss.item():.4f}")

        total_loss += loss.item()
        total_acc += (logits.argmax(dim=1) == labels).float().mean()

    # tensorboard跟踪
    writer.add_scalar("val_avg_loss", total_loss / len(val_dl), val_loss_cnt)
    val_loss_cnt += 1

    # 计算准确率
    writer.add_scalar("val_accuracy", total_acc / len(val_dl), val_acc_cnt)
    val_acc_cnt += 1


def validate_step(model, tokens, labels, criterion):
    with torch.no_grad():  # 禁用梯度计算[7,8](@ref)
        logits = model(tokens)  # 前向传播[6](@ref)
        loss = criterion(logits, labels)  # 计算损失[6](@ref)
        return loss, logits  # 返回损失和预测值[8](@ref)


if __name__ == "__main__":
    # tensorboard 跟踪
    # 0.创建SummaryWriter对象
    writer = SummaryWriter(log_dir="runs/hw1_comment_classification")

    # 跟踪相关变量
    train_loss_cnt = 0
    val_loss_cnt = 0
    val_acc_cnt = 0

    # 超参数
    BATCH_SIZE = 128  # 批大小
    EPOCH = 10  # 训练轮数
    EMBEDDING_SIZE = 200  # 词向量维度
    RNN_HIDDEN_SIZE = 200  # RNN隐藏层维度
    LEARN_RATE = 0.001  # 学习率
    NUM_LABELS = 2  # 标签数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.加载数据
    with open(f"{current_dir}/comments_spm.bin", "rb") as f:
        comments_jieba, labels = pickle.load(f)
    vacab = build_vocab_from_documents(comments_jieba)

    # 2.数据拆分
    (x_train, y_train), (x_val, y_val) = train_test_split(comments_jieba, labels)
    (x_valid, y_valid), (x_test, y_test) = train_test_split(
        x_val, y_val, split_rate=0.5
    )

    # 3.构建DataLoader
    train_ds = list(zip(x_train, y_train))
    tarin_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        collate_fn=build_collate_func(vocab=vacab),
        shuffle=True,
    )

    valid_ds = list(zip(x_valid, y_valid))
    valid_dl = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        collate_fn=build_collate_func(vocab=vacab),
        shuffle=True,
    )

    test_ds = list(zip(x_test, y_test))
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        collate_fn=build_collate_func(vocab=vacab),
        shuffle=False,
    )

    # 4.构建模型
    model = CommentsClassifier(
        vocab_size=len(vacab),
        emb_size=EMBEDDING_SIZE,
        rnn_hidden_size=RNN_HIDDEN_SIZE,
        num_labels=NUM_LABELS,
    ).to(device)

    # 5.定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    criterion = nn.CrossEntropyLoss()

    # 6.训练和验证
    for epoch in range(EPOCH):
        train(model, tarin_dl, criterion, optimizer)
        validate(model, valid_dl, criterion)

    # 模型保存
    torch.save(
        {"model_state": model.state_dict(), "model_vocab": vacab},
        f"{current_dir}/model_objs2.bin",
    )
