'''
1.获取图书字典，key为书名，value为每本书的评论的分词列表
2.使用bm25算法，计算相似度，得到相似度矩阵，每一行是一个书名的相似度
3.测试方法，输入书名，输出相似度最高的10本书名和相似度
'''
import csv
import jieba
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi



def load_data(filename):
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if book == '':
                continue
            book_comments[book] = book_comments.get(book, [])
            book_comments[book].extend(jieba.lcut(comment))
    return book_comments

def bm25(book_comment):
    bm25 = BM25Okapi(book_comment)
    bm25_mitrix = []
    for i in range(len(book_comment)):
        bm25_mitrix.append(bm25.get_scores(book_comment[i]))
    return cosine_similarity(bm25_mitrix)

def test_bm25():
    book_comments = load_data('douban_bookcomments.txt')
    book_names = []
    book_common = []
    for book, comments in book_comments.items():
        book_names.append(book)
        book_common.append(comments)

    bm25_mitrix = bm25(book_common)
    print(book_names)
    book_name = input("请输入书名:")
    book_index = book_names.index(book_name)
    similar_books = np.argsort(bm25_mitrix[book_index])[::-1][1:11]
    for index in similar_books:
        print("书名：",book_names[index], "相似度：", bm25_mitrix[book_index][index])

if __name__ == '__main__':
    test_bm25()