import os

from collections import defaultdict


def concatenate_txt_files(folder_path):
    concatenated_text = ""
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    concatenated_text += content + "\n"
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='gbk') as file:
                        content = file.read()
                        concatenated_text += content + "\n"
                except Exception as e:
                    print(f"read file failed {filename}: {e}")
            except Exception as e:
                print(f"read file failed {filename}: {e}")
    
    return concatenated_text


def bpe(str, max_num_word=50):
    vocab = {}
    current_index = 0
    
    for char in set(str):
        vocab[current_index] = char
        current_index += 1

    s = [2] * (len(str) - 1) + [0] # pos is token length, neg is word length
    pair_to_loc = defaultdict(list) # {"token": [location1, location2, location3, ... ]}

    for i in range(len(str)-1):
        pair_to_loc[str[i:i+2]].append(i)
    
    while len(vocab) < max_num_word:
        new_token, index_list = max(pair_to_loc.items(), key=lambda x: len(x[1]))
        if len(index_list) == 1:
            break

        vocab[current_index] = new_token
        
        del pair_to_loc[new_token]
        for id in index_list:
            new_word_len = s[id]
            if new_word_len < 0:
                continue
            if s[id+1] < 0:
                if s[id-s[id+1]] != 0:
                    if str[id-s[id+1]:id-s[id+1]+s[id-s[id+1]]] in pair_to_loc:
                        pair_to_loc[str[id-s[id+1]:id-s[id+1]+s[id-s[id+1]]]].remove(id-s[id+1])
                    s[id] = s[id-s[id+1]] - s[id+1]
                    s[id-s[id+1]] = 0
                    s[id+1] = -new_word_len
                    s[id+new_word_len-1] = -new_word_len
                    pair_to_loc[str[id:id+s[id]]].append(id)
                else:
                    s[id] = 0
                    s[id-s[id+1]-1] = 0
                    s[id+1] = 0
            elif s[id+1] > 0:
                if str[id+1:id+1+s[id+1]] in pair_to_loc:
                    pair_to_loc[str[id+1:id+1+s[id+1]]].remove(id+1)
                s[id] = s[id+1] + 1
                s[id+1] = -new_word_len
                s[id+new_word_len-1] = -new_word_len
                pair_to_loc[str[id:id+s[id]]].append(id)
            else:
                s[id] = 0
            
            if id > 0:
                if s[id-1] < 0:
                    f_token_id = id+s[id-1]
                    f_token_len_new = new_word_len - s[id-1]
                else:
                    f_token_id = id - 1
                    f_token_len_new = new_word_len + 1

                f_token = str[f_token_id:f_token_id+s[f_token_id]]
                f_list = pair_to_loc[f_token]
                if f_token_id in f_list:
                    f_list.remove(f_token_id)
                if len(f_list) == 0:
                    del pair_to_loc[f_token]
                pair_to_loc[str[f_token_id:f_token_id+f_token_len_new]].append(f_token_id)
                s[f_token_id] = f_token_len_new
            
        current_index += 1
    
    tokenized_str = []
    i = 0
    while True:
        if i == len(s) - 1:
            tokenized_str.append(str[i])
            break

        if s[i+1] > 0:
            tokenized_str.append(str[i])
            i += 1
        elif s[i+1] < 0:
            tokenized_str.append(str[i:i-s[i+1]])
            i -= s[i+1]
        else:
            tokenized_str.append(str[i:])
            break

    return vocab, tokenized_str


def validate_bpe(vocab, tokenized_str, original_str):
    vocab_values = set(vocab.values())
    for token in tokenized_str:
        if token not in vocab_values:
            print(f"Validation Error: Token '{token}' not in vocab")
            return False
    
    reconstructed = ''.join(tokenized_str)
    if reconstructed != original_str:
        print(f"Validation Error: Mismatch in string reconstruction")
        print(f"Original: {original_str}")
        print(f"Reconstructed: {reconstructed}")
        return False
    
    return True


def indices_to_string(indices, vocab):
    return ''.join(vocab.get(idx, 'UNK') for idx in indices)


def tokenize_with_vocab(text, vocab):
    vocab_set = set(vocab.values())
    max_token_len = max(len(token) for token in vocab_set) if vocab_set else 1
    
    tokens = []
    i = 0
    n = len(text)
    
    while i < n:
        matched = False
        for l in range(min(max_token_len, n-i), 0, -1):
            candidate = text[i:i+l]
            if candidate in vocab_set:
                tokens.append(candidate)
                i += l
                matched = True
                break
        
        if not matched:
            if text[i] in vocab_set:
                tokens.append(text[i])
            else:
                tokens.append('UNK')
            i += 1
    
    return tokens


if __name__ == "__main__":
    str = concatenate_txt_files(r"D:\desktop\.test\AiPremiumClass\邪王真眼\datasets\heroes")
    max_num_word = 3000
    vocab, tokenized_str = bpe(str, max_num_word)

    is_valid = validate_bpe(vocab, tokenized_str, str)
    print(f"Validation Result: {'Success' if is_valid else 'Failed'}")
    print(f"Oringal string length: {len(str)}\t BPE zipped length: {len(tokenized_str)}")
    sample_indices = [0, 1, 2]
    print(f"Sample indices to string: {indices_to_string(sample_indices, vocab)}")
    
    new_text = "The cat sings UNKNOWN"
    new_tokens = tokenize_with_vocab(new_text, vocab)
    print(f"Tokenized new text: {new_tokens}")
