import torch
import pickle
from couplets import Encoder, Decoder, Seq2Seq, Vocabulary


def load_model_and_vocab(model_path, vocab_path, config):
    """加载训练好的模型和词典"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载词汇表
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # 初始化模型
    model = Seq2Seq(
        enc_emb_size=len(vocab.word2idx),
        dec_emb_size=len(vocab.word2idx),
        emb_dim=config.emb_dim,
        hidden_size=config.hidden_dim,
        dropout=config.dropout
    ).to(device)

    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, vocab, device


def predict(model, vocab, device, input_sentence, max_len=50):
    """模型推理函数"""
    # 将输入句子转换为索引序列
    model.eval()
    input_indices = [[vocab.word2idx.get(char, vocab.word2idx['<unk>'])
                      for char in input_sentence]]

    # 转换为tensor并移动到设备
    input_tensor = torch.tensor(input_indices, dtype=torch.long).to(device)

    with torch.no_grad():
        # 编码器前向传播
        encoder_state, enc_outputs = model.encoder(input_tensor)

        hidden_state = encoder_state# batch, hidden_dim*2]

        # 初始化解码器输入（<bos>）
        decoder_input = torch.tensor([[vocab.word2idx['<bos>']]], device=device)

        # 存储解码结果
        decoded_chars = []

        for _ in range(max_len):
            # 解码器前向传播
            output, hidden_state = model.decoder(decoder_input, hidden_state, enc_outputs)

            hidden_state = hidden_state.squeeze(0)

            # 获取预测的下一个字符
            topi = output.argmax(dim=-1)  # 取最后一个时间步
            decoded_char = vocab.idx2word[topi.item()]

            # 遇到<eos>则停止
            if decoded_char == '<eos>':
                break

            decoded_chars.append(decoded_char)

            # 下一个解码器输入是当前预测的字符
            decoder_input = topi

    return ''.join(decoded_chars)


if __name__ == '__main__':
    # 配置参数（需要与训练时一致）
    class InferenceConfig:
        emb_dim = 100
        hidden_dim = 120
        dropout = 0.5


    # 加载模型和词典
    model, vocab, device = load_model_and_vocab(
        model_path='seq2seq_model.pt',
        vocab_path='vocab.pkl',
        config=InferenceConfig()
    )

    # 用户输入上联
    while True:
        upper_couplet = input("请输入上联（输入q退出）: ").strip()
        if upper_couplet.lower() == 'q':
            break

        # 生成下联
        lower_couplet = predict(model, vocab, device, upper_couplet,max_len=len(upper_couplet))
        print(f"上联: {upper_couplet}")
        print(f"下联: {lower_couplet}")
        print("-" * 50)