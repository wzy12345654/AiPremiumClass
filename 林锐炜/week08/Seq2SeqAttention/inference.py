import pickle

import torch

from model import Seq2Seq


class CoupletGenerator:
    def __init__(self, config_path='config.pkl', model_path='seq2seq_couplet.pt', max_length=50):
        # 加载配置
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)

        # 设备配置
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        # 加载词汇表
        with open(self.config.enc_vocab_path, 'rb') as f:
            self.enc_vocab = pickle.load(f)
        with open(self.config.dec_vocab_path, 'rb') as f:
            self.dec_vocab = pickle.load(f)

        # 初始化模型
        self.model = Seq2Seq(
            config=self.config,
            enc_vocab=self.enc_vocab,
            dec_vocab=self.dec_vocab
        ).to(self.device)

        # 加载训练权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.max_length = max_length
        self.dec_bos_idx = self.dec_vocab.stoi[self.dec_vocab.bos]
        self.dec_eos_idx = self.dec_vocab.stoi[self.dec_vocab.eos]

    def _preprocess(self, input_str):
        """将输入字符串转换为编码张量"""
        tokens = list(input_str.strip())
        ids = [self.enc_vocab.stoi.get(token, self.enc_vocab.stoi[self.enc_vocab.unk])
               for token in tokens]
        return torch.LongTensor(ids).unsqueeze(0).to(self.device)  # [1, seq_len]

    def _postprocess(self, token_ids):
        """将token序列转换为字符串"""
        return ''.join(
            self.dec_vocab.itos[idx]
            for idx in token_ids
            if idx not in {self.dec_bos_idx, self.dec_eos_idx}
        )

    def generate(self, input_str):
        """对联生成主函数"""
        with torch.no_grad():
            # 1. 编码输入
            src = self._preprocess(input_str)

            # 2. Encoder前向传播
            encoder_outputs, hidden = self.model.encoder(src)

            # 3. 准备Decoder初始输入
            dec_input = torch.tensor([[self.dec_bos_idx]], device=self.device)  # [1, 1]
            output_seq = []

            # 4. 自回归解码
            for _ in range(self.max_length):
                output, hidden = self.model.decoder(
                    dec_input,  # [1, 1]
                    hidden,  # [1, hidden_size]
                    encoder_outputs  # [1, src_len, hidden_size*2]
                )

                # 获取预测token
                pred_token = output.argmax(-1).item()
                output_seq.append(pred_token)

                # 遇到EOS停止
                if pred_token == self.dec_eos_idx:
                    break

                # 准备下一步输入
                dec_input = torch.tensor([[pred_token]], device=self.device)

            return self._postprocess(output_seq)


if __name__ == "__main__":
    # 使用示例
    generator = CoupletGenerator(
        config_path="config.pkl",  # 保存的配置文件
        model_path="seq2seq_couplet.pt",
        max_length=50
    )

    test_cases = [
        "春风暖万家",
        "书山有路勤为径",
        "天增岁月人增寿",
        "处处春光好"
    ]

    for case in test_cases:
        result = generator.generate(case)
        print(f"上联：{case}")
        print(f"下联：{result}\n")
