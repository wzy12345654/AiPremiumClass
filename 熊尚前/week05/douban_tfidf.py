import csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_data(filename):
    """
    从指定文件中加载数据。

    :param filename: 包含数据的文件路径
    :return: 无（当前函数未完成返回逻辑）
    """
    # 图书评论集合 {书名： "评论1" + "评论2" + ...}
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for item in reader:
            book = item['book']
            comment_words = jieba.lcut(item['body'])
            
            if book == '':
                continue

            # 图书评论集合收集
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(comment_words)
    return book_comments


def tf_idf(book_comments):

    tf_idf_vectorizer = TfidfVectorizer()
    tf_idf_matrix = tf_idf_vectorizer.fit_transform([' '.join(commons) for commons in book_comments])
    # 计算图书的余弦相似度
    cosine_similarities = cosine_similarity(tf_idf_matrix)
    return cosine_similarities


def test_tfidf():
    # 加载图书评论数据
    book_comments = load_data('douban_bookcomments.txt')

    # 提取书名和评论文本
    book_names = []
    book_common = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_common.append(comments)

    # 计算TF-IDF矩阵
    tfidf_matrix = tf_idf(book_common)

    print(book_names)
    book_name = input("请输入书名:")
    book_index = book_names.index(book_name)
    similar_books = np.argsort(tfidf_matrix[book_index])[::-1][1:11]
    for index in similar_books:
        print("书名：",book_names[index], "相似度：", tfidf_matrix[book_index][index])

if __name__ == '__main__':
    test_tfidf()