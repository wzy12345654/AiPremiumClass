import tensorflow as tf
import pandas as pd
import numpy as np
import datetime

from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

#1. 使用中文对联数据集训练带有attention的seq2seq模型，利用tensorboard跟踪
# 读取数据集：每两行组成一对 input/output
data = []
with open('vocabs', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]  # 去除空行

    # 确保总行数是偶数，如果不是，可以去掉最后一个不完整的项
    if len(lines) % 2 != 0:
        lines = lines[:-1]

    # 每两行组成一个对
    for i in range(0, len(lines), 2):
        input_text = lines[i]
        output_text = lines[i+1] if i+1 < len(lines) else ''
        data.append({
            'input': input_text,
            'output': output_text
        })

# 转换为 DataFrame
df = pd.DataFrame(data)

# 分割数据集
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

#构建词汇表
def build_vocab(sentences, vocab_size):
    # 统计每个单词出现的次数
    word_counts = Counter(word for sentence in sentences for word in sentence.split())
    
    # 添加特殊标记
    special_tokens = ['<start>', '<end>', '<pad>']
    vocab = special_tokens + [word for word, _ in word_counts.most_common(vocab_size - len(special_tokens))]
    
    # 构建词到索引的映射
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    
    return word_to_index, index_to_word

vocab_size = 10000
input_word_to_index, input_index_to_word = build_vocab(train_data['input'], vocab_size)
output_word_to_index, output_index_to_word = build_vocab(train_data['output'], vocab_size)

# 定义一个函数，用于将句子编码为索引
def encode_sentences(sentences, word_to_index, max_length):
    encoded_sentences = []
    for sentence in sentences:
        words = sentence.split()
        # 在句子末尾添加 <end>
        encoded_sentence = [word_to_index.get(word, 0) for word in words] + [word_to_index['<end>']]
        # 截断或补零
        encoded_sentence = encoded_sentence[:max_length] + [0] * (max_length - len(encoded_sentence))
        encoded_sentences.append(encoded_sentence)
    return np.array(encoded_sentences)

max_length = 20
train_input = encode_sentences(train_data['input'], input_word_to_index, max_length)
train_output = encode_sentences(train_data['output'], output_word_to_index, max_length)
test_input = encode_sentences(test_data['input'], input_word_to_index, max_length)
test_output = encode_sentences(test_data['output'], output_word_to_index, max_length)


embedding_dim = 256

units = 512

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

# Attention
attention = tf.keras.layers.Attention()
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

# Dense layer
output = Dense(vocab_size, activation='softmax')(decoder_combined_context)

model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')



log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# model.fit([train_input, train_output[:, :-1]], np.expand_dims(train_output[:, 1:], -1),
#           batch_size=64,
#           epochs=10,
#           validation_data=([test_input, test_output[:, :-1]], np.expand_dims(test_output[:, 1:], -1)),
#           callbacks=[tensorboard_callback])

#2. 尝试encoder hidden state不同的返回形式（concat和add）
# Concatenate encoder and decoder states
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

# Add encoder and decoder states
context_vector = attention([decoder_outputs, encoder_outputs])
decoder_combined_context = tf.keras.layers.Add()([context_vector, decoder_outputs])

#3. 编写并实现seq2seq attention版的推理实现
# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(units,))
decoder_state_input_c = Input(shape=(units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
context_vector2 = attention([decoder_outputs2, encoder_outputs])
decoder_combined_context2 = Concatenate(axis=-1)([context_vector2, decoder_outputs2])
output2 = Dense(vocab_size, activation='softmax')(decoder_combined_context2)

# 修改 decoder_model 的定义，只返回输出和状态，不包含注意力权重
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs + [encoder_outputs],
    [output2, state_h2, state_c2]  # 注意：这里去掉了 attention_weights
)

# Inference


def decode_sequence(input_seq):
    # 使用 encoder 模型获取 encoder_outputs 和初始状态
    encoder_model_inf = Model(encoder_inputs, [encoder_outputs] + encoder_states)
    encoder_outs, state_h, state_c = encoder_model_inf.predict(input_seq)

    # 初始化目标序列，第一个词为 '<start>'
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_word_to_index['<start>']

    # 初始化状态值
    states_value = [state_h, state_c]

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # 调用 decoder 模型进行预测
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value + [encoder_outs]
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])

        # ⬇️ 添加边界检查，防止 KeyError
        if sampled_token_index < 0 or sampled_token_index >= vocab_size:
            sampled_token_index = output_word_to_index['<pad>']  # 或者使用 '<unk>' 占位符

        sampled_word = output_index_to_word.get(sampled_token_index, '<unk>')
        decoded_sentence += ' ' + sampled_word

        if sampled_word == '<end>' or len(decoded_sentence.split()) >= max_length:
            stop_condition = True

        # 更新目标序列和状态
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()

# Test
for seq_index in range(10):
    input_seq = train_input[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', train_data.iloc[seq_index]['input'])
    print('Decoded sentence:', decoded_sentence)