import torch
import jieba
import text_classifier

if __name__ == '__main__':
    # 加载词典
    vocab = torch.load('model/comments_vocab.pth')
    # 构建模型
    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 2  # 二分类

    # 加载模型
    model = text_classifier.Comments_Classfier(vocab_size, embedding_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('model/text_classifier.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    comment1 = "这部电影真好看！"
    comment2 = "这部电影的演员演技真差！"

    # 将评论转换为索引
    comment1_idx = [vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment1)]
    comment2_idx = [vocab.get(word, vocab['UNK']) for word in jieba.lcut(comment2)]

    # 转换为tensor
    comment1_tensor = torch.tensor(comment1_idx, dtype=torch.long).unsqueeze(0).to(device)  # 添加batch维度
    comment2_tensor = torch.tensor(comment2_idx, dtype=torch.long).unsqueeze(0).to(device)  # 添加batch维度

    # 模型预测
    model.eval()
    with torch.no_grad():
        output1 = model(comment1_tensor)
        output2 = model(comment2_tensor)
        _, predicted1 = torch.max(output1, 1)
        _, predicted2 = torch.max(output2, 1)
        print(f"Comment 1: {comment1} => Predicted class: {predicted1.item()}")
        print(f"Comment 2: {comment2} => Predicted class: {predicted2.item()}")
    # 1: POSITIVE, 0: NEGATIVE

