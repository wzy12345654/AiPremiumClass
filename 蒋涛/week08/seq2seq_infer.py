"""
1. 加载训练好模型和词典
2. 解码推理流程
    - 用户输入通过vocab转换token_index
    - token_index通过encoder获取 encoder last hidden_state
    - 准备decoder输入第一个token_index:[['BOS']] shape: [1,1]
    - 循环decoder
        - decoder输入:[['BOS']], hidden_state
        - decoder输出: output,hidden_state  output shape: [1,1,dec_voc_size]
        - 计算argmax, 的到下一个token_index
        - decoder的下一个输入 = token_index
        - 收集每次token_index 【解码集合】
    - 输出解码结果
"""
import torch
import pickle
import random
from process import read_data, get_proc, Vocabulary
from EncoderDecoderAttenModel import Seq2Seq

def test_model():
    """
    对训练好的 Seq2Seq 模型进行测试推理。
    随机选取测试样本，使用训练好的模型生成下联，并与真实下联进行对比。
    """
    # 定义测试数据文件路径
    enc_test_file = 'week8_codes/couplet/test/in.txt'  # 上联测试数据文件路径
    dec_test_file = 'week8_codes/couplet/test/out.txt'  # 下联测试数据文件路径

    # 读取测试数据
    enc_data, dec_data = read_data(enc_test_file, dec_test_file)

    # 加载训练好的模型和词典
    state_dict = torch.load('seq2seqadd_state.bin')  # 加载模型的状态字典
    vocab_file = 'week8_codes/couplet/vocabs'  # 词汇表文件路径
    vocab = Vocabulary.from_file(vocab_file)  # 从文件中加载词汇表

    # 初始化 Seq2Seq 模型
    model = Seq2Seq(
        enc_emb_size=len(vocab.vocab),  # 编码器嵌入层的输入维度，即词汇表大小
        dec_emb_size=len(vocab.vocab),  # 解码器嵌入层的输入维度，即词汇表大小
        emb_dim=200,  # 嵌入层的维度
        hidden_size=250,  # 隐藏层的维度
        dropout=0.5,  # Dropout 概率，防止过拟合
        state_type='add',  # 编码器隐藏状态的处理方式
    )
    model.load_state_dict(state_dict)  # 将加载的状态字典应用到模型上

    # 创建解码器反向字典，用于将 token 索引转换为对应的字符
    dvoc_inv = {v: k for k, v in vocab.vocab.items()}

    # 随机选取测试样本
    rnd_idx = random.randint(0, len(enc_data) - 1)  # 生成一个随机索引
    enc_input = enc_data[rnd_idx]  # 根据随机索引获取上联输入
    dec_output = dec_data[rnd_idx]  # 根据随机索引获取真实下联输出

    # 将上联输入转换为 token 索引张量
    enc_idx = torch.tensor([[vocab.vocab[tk] for tk in enc_input]])

    print(enc_idx.shape)  # 打印上联输入的张量形状

    # 推理
    # 最大解码长度，设置为上联的长度
    max_dec_len = len(enc_input)

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 不计算梯度，加快推理速度
        # 编码器输出
        hidden_state, enc_outputs = model.encoder(enc_idx)  # 调用编码器获取隐藏状态和输出

        # 解码器输入，初始化为起始标记 <s> 的 token 索引
        dec_input = torch.tensor([[vocab.vocab['<s>']]])

        # 循环解码
        dec_tokens = []  # 用于存储解码得到的 token 索引
        while True:
            if len(dec_tokens) >= max_dec_len:  # 如果解码长度达到最大长度，停止解码
                break
            # 解码器
            # logits 形状: [1, 1, dec_voc_size]
            logits, hidden_state = model.decoder(dec_input, hidden_state, enc_outputs)

            # 获取下个 token 的索引
            next_token = torch.argmax(logits, dim=-1)

            if dvoc_inv[next_token.squeeze().item()] == '</s>':  # 如果遇到结束标记 </s>，停止解码
                break
            # 收集每次解码得到的 token 索引
            dec_tokens.append(next_token.squeeze().item())
            # 将下个 token 索引作为下一次解码器的输入
            dec_input = next_token
            hidden_state = hidden_state.view(1, -1)  # 调整隐藏状态的形状

    # 输出解码结果
    print(f'上联：', ''.join(enc_input))  # 打印上联
    print("模型预测下联：", ''.join([dvoc_inv[tk] for tk in dec_tokens]))  # 打印模型预测的下联
    print("真实下联：", ''.join(dec_output))  # 打印真实下联

    
if __name__ == '__main__':
    test_model()