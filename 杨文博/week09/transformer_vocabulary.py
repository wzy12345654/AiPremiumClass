import pickle


class Config:
    upper_file_path = './in.txt'
    lower_file_path = './out.txt'
    batch_size = 256
    dropout = 0.5
    max_seq_len = 64
    embedding_dim = 100
    num_heads = 4
    num_layers = 2
    epoches = 20
    learning_rate = 1e-4


class Vocabulary:
    def __init__(self):
        self.upper_couplets = []
        self.lower_couplets = []
        self.vocab = set()
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}

    @staticmethod
    def __load_data(data_path):
        with open(data_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        return [
            [char for char in line.strip() if char != ' ']  # 过滤空格
            for line in lines
        ]

    def load_couplets_data(self):
        self.upper_couplets = self.__load_data(Config.upper_file_path)
        self.lower_couplets = self.__load_data(Config.lower_file_path)

    def get_vocabulary(self):
        for upper in self.upper_couplets:
            self.vocab.update(upper)
        for lower in self.lower_couplets:
            self.vocab.update(lower)

    def get_two_dicts(self):
        self.word2idx.update({char: idx + 4 for idx, char in enumerate(self.vocab)})
        self.idx2word.update({idx + 4: char for idx, char in enumerate(self.vocab)})

    def save_vocabulary(self):
        with open('./vocab.pkl', 'wb') as f:
            pickle.dump(self, f)

    def get_vocab_and_save(self):
        self.load_couplets_data()
        self.get_vocabulary()
        self.get_two_dicts()
        self.save_vocabulary()

if __name__ == '__main__':
    vocab = Vocabulary()
    vocab.get_vocab_and_save()