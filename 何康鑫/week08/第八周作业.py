import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard
import os

# 参数配置
BUFFER_SIZE = 20000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
UNITS = 1024
VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 30
ATTENTION_FEATURE_MAPS = 32

# 数据准备
def load_data(data_path):
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                texts.append(line)
    return texts

# 加载数据集
train_texts = load_data('chinese-couplets/train.txt')
test_texts = load_data('chinese-couplets/test.txt')

# 创建Tokenizer
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts + test_texts)

# 序列化文本
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# 填充序列
train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# 创建输入输出对（下联作为目标，左移一位）
def create_dataset(sequences):
    X, y = [], []
    for seq in sequences:
        X.append(seq[:-1])
        y.append(seq[1:])
    return np.array(X), np.array(y)

train_X, train_y = create_dataset(train_padded)
test_X, test_y = create_dataset(test_padded)

# 构建注意力机制层
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# 定义Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)
encoder_lstm = LSTM(UNITS, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 定义Decoder（两种hidden state处理方式）
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
decoder_lstm_concat = LSTM(UNITS*2, return_sequences=True, return_state=True)
decoder_lstm_add = LSTM(UNITS, return_sequences=True, return_state=True)

# 注意力机制整合
attention = BahdanauAttention(ATTENTION_FEATURE_MAPS)

# Decoder两种模式
def build_decoder(decoder_lstm):
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    context_vector, attention_weights = attention(decoder_outputs, encoder_outputs)
    
    # 输出层
    output = Concatenate()([context_vector, decoder_outputs])
    output = Dense(VOCAB_SIZE, activation='softmax')(output)
    return Model([decoder_inputs] + encoder_states, [output] + encoder_states)

# 构建两个Decoder模型（concat和add）
decoder_concat = build_decoder(decoder_lstm_concat)
decoder_add = build_decoder(decoder_lstm_add)

# 训练模型（以concat为例）
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        predictions, _, _ = decoder_concat([inp] + enc_hidden)
        loss = loss_function(targ, predictions)
    gradients = tape.gradient(loss, decoder_concat.trainable_variables)
    optimizer.apply_gradients(zip(gradients, decoder_concat.trainable_variables))
    train_loss(loss)
    train_accuracy(targ, predictions)

# 训练过程
EPOCHS = 10
tensorboard_callback = TensorBoard(log_dir="logs/{}".format(time.time()))

for epoch in range(EPOCHS):
    start = time.time()
    
    # 初始化隐藏状态
    enc_hidden = encoder_lstm.initialize_states(BATCH_SIZE)
    
    for (batch, (inp, targ)) in enumerate(dataset.take(len(train_X)//BATCH_SIZE)):
        train_step(inp, targ, enc_hidden)
        
    print(f'Epoch {epoch+1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start:.2f} sec\n')

# 推理实现
def evaluate(sentence, decoder, encoder_model, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    attention_plot = np.zeros((max_length, max_length))
    
    sentence = preprocess_sentence(sentence)
    inputs = tokenizer.texts_to_sequences([sentence])
    inputs = pad_sequences(inputs, maxlen=max_length, padding='post')
    
    states_value = encoder_model.predict(inputs)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word[sampled_token_index]
        decoded_sentence += ' ' + sampled_char
        
        if (sampled_char == '<end>' or 
            len(decoded_sentence.split()) > max_length):
            stop_condition = True
            
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        states_value = [h, c]
        
    return decoded_sentence

# 使用示例
encoder_model = Model(encoder_inputs, encoder_states)
print(evaluate("花开富贵", decoder_concat, encoder_model, tokenizer))
print(evaluate("花开富贵", decoder_add, encoder_model, tokenizer))
