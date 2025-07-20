import torch
import pickle
from transformer_model import Seq2SeqTransformer, generate_square_subsequent_mask

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('vocabs.bin','rb') as f:
        evoc,dvoc = pickle.load(f)
    # 构建逆向词表：id -> token
    dvoc_inv = {v: k for k, v in dvoc.items()}

    model = Seq2SeqTransformer(
        d_model=256,
        nhead=4,
        num_enc_layers=2,
        num_dec_layers=2,
        dim_forward=512,
        dropout=0.1,
        enc_voc_size=len(evoc),
        dec_voc_size=len(dvoc)
    )
    model.to(device)
    state_dict = torch.load('transformer_final.bin', map_location='cpu')  # 加载到CPU上
    model.load_state_dict(state_dict)
    model.eval()
    print("模型已加载完成")

    while True:
        input_text = input("请输入内容：")
        if input_text.lower() in  ['exit','quit','退出']:
            print("再见！")
            break

    # 将输入编码为 token index 序列
        enc_input = [evoc.get(tk, evoc['UNK']) for tk in input_text]
        enc_tensor = torch.tensor([enc_input], dtype=torch.long)  # batch_size=1
        enc_tensor  = enc_tensor.to(device)
        # 使用 encode 方法获取 memory
        with torch.no_grad():
            memory = model.encode(enc_tensor)

        # 自回归解码生成
        max_len = 20  # 最大生成长度
        bos_id = dvoc.get('<s>', None)
        eos_id = dvoc.get('</s>', None)
        if bos_id is None or eos_id is None:
            raise ValueError("词表中未找到 <s> 或 </s> 标记，请确保训练数据包含这些符号。")

        # 初始 decoder 输入为 <s>
        dec_input = [bos_id]

        for _ in range(max_len):
            dec_tensor = torch.tensor([dec_input], dtype=torch.long)
            tgt_mask = generate_square_subsequent_mask(dec_tensor.size(1)).to(dec_tensor.device)

            with torch.no_grad():
                output = model.decode(dec_tensor, memory, tgt_mask)

            # 预测下一个词
            logits = model.predict(output[:, -1, :])
            next_token_id = logits.argmax(dim=-1).item()

            # 如果预测到结束符，则停止
            if next_token_id == eos_id:
                break

            dec_input.append(next_token_id)

        # 解码输出序列
        generated_text = ''.join([dvoc_inv[tid] for tid in dec_input[1:]])  # 跳过 <s>
        print(f"生成结果: {generated_text}")
