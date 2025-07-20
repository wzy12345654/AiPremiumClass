from ast import main
import csv
from functools import lru_cache     # 内存中的缓存
from joblib import Memory           # 磁盘缓存
from tkinter.tix import MAIN
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 定义磁盘缓存目录
memory = Memory(location='./cache_directory', verbose=0)

@memory.cache
def load_data(fname):
    print("loading......")
    book_comments = {}

    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")

        # 生成词向量练习文档
        # comments_text = open('comments.txt', 'w+', encoding='utf-8')
        for item in reader:
            book_name = item['book']
            book_comment = item['body']
            if book_comment is not None: 
                comment_words = jieba.lcut(book_comment)
                # comments_text.write(' '.join(comment_words) + '\n')
            book_comments[book_name] = book_comments.get(book_name, [])
            book_comments[book_name].extend(comment_words)
    return book_comments

def bm25(comments, k=1.5, b=0.75):
    # 计算文档总数
    N = len(comments)
    # 初始化文档列表和词频字典
    doc_len = []
    word_doc_freq = {}
    doc_term_dict = [{} for _ in range(N)]

    for i, comment in enumerate(comments.values()):
        # 记录文档长度
        doc_len.append(len(comment))
        unique_words = set()
        for word in comment:
            # 统计词频
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        # 统计包含该词的文档数量
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
        
    # 计算每个单词的平均文档长度
    avg_doc_len = sum(doc_len) / N

    # 构建词汇表
    vocabulary = list(word_doc_freq.keys())
    word_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # 构建文档-词频矩阵
    doc_term_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        for word, freq in doc_term_dict[i].items():
            idx = word_idx.get(word)
            if idx is not None:
                doc_term_matrix[i, idx] = freq
    
    # 计算idf值
    idf_numerator = N - np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    idf_denominator = np.array([word_doc_freq[word] for word in vocabulary]) + 0.5
    # idf_numerator = N - idf_denominator + 1
    idf = np.log(idf_numerator / idf_denominator)
    idf[idf_numerator <= 0] = 0 # 避免出现nan值

    # 计算bm25值
    doc_len = np.array(doc_len)
    bm25_matrix = np.zeros((N, len(vocabulary)))
    for i in range(N):
        tf = doc_term_matrix[i]
        bm25 = idf * (tf * (k + 1)) / (tf + k * (1 - b + b * doc_len[i] / avg_doc_len))
        bm25_matrix[i] = bm25

    # 根据原始评论顺序重新排列bm25值
    final_bm25_matrix = []
    for i, comment in enumerate(comments.values()):
        bm25_comment = []
        for word in comment:
            idx = word_idx.get(word)
            if idx is not None:
                bm25_comment.append(bm25_matrix[i, idx])
        final_bm25_matrix.append(bm25_comment)
    
    # 找到最长的子列表长度
    max_length = max(len(row) for row in final_bm25_matrix)
    # 填充所有子列表到相同长度
    padded_matrix = [row + [0] * (max_length - len(row)) for row in final_bm25_matrix]
    # 转换为numpy数组
    final_bm25_matrix = np.array(padded_matrix)

    return final_bm25_matrix

if __name__ == '__main__':

    stop_words = [line.strip() for line in open('./stop_words.txt', 'r', encoding='utf-8')]
    book_comments = load_data("book_comments_fix.txt")

    book_name_list = []
    book_comments_list = []
    for book_name, book_comment in book_comments.items():
        book_name_list.append(book_name)
        book_comments_list.append(book_comments)

    # 构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform([' '.join(comments) for comments in book_comments.values()])
    # print(tfidf_matrix.shape)

    # bm25矩阵
    bm25_matrix = bm25(book_comments)

    similarity_matrix_tfidf = cosine_similarity(tfidf_matrix)
    similarity_matrix_bm25 = cosine_similarity(bm25_matrix)
    # print(similarity_matrix.shape)

    print(book_name_list)
    book_name_input = input("输入图书名称: ")
    book_idx = book_name_list.index(book_name_input)    # 获取图书索引

    book_idxs = np.argsort(-similarity_matrix_tfidf[book_idx])    # 根据索引获取对应行，并排序（argsort默认升序，所以加-号）
    recommendation_top10_idxs = book_idxs[1:11]     # 取第一条数据后的十条数据

    print("------------------tfidf算法推荐------------------------")
    for idx in recommendation_top10_idxs:
        book_name = book_name_list[idx]
        print(f'tfidf推荐: \n《{book_name}》,\t 相似度: {similarity_matrix_tfidf[book_idx][idx]}')

    book_idxs2 = np.argsort(-similarity_matrix_bm25[book_idx])    # 根据索引获取对应行，并排序（argsort默认升序，所以加-号）
    recommendation_top10_idxs2 = book_idxs[1:11]     # 取第一条数据后的十条数据

    print("------------------bm25算法推荐------------------------")
    for idx in recommendation_top10_idxs2:
        book_name = book_name_list[idx]
        print(f'{book_name}》,\t 相似度: {similarity_matrix_bm25[book_idx][idx]}')