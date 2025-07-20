import  csv
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np


def load_data(filename):
#图书评论集合
    book_conments = {} #{'book1': ['comment1', 'comment2', ...], 'book2': ['comment1', 'comment2', ...], ...}
    with open(filename, 'r', encoding='utf-8') as f:
        #.DictReader()方法可以识别文本中的标题列
        reader = csv.DictReader(f, delimiter='\t')  #识别格式文本中的标题列
        for item in reader:
            book = item['book']
            comment = item['body']
            #jieba.lcut()方法可以将文本分词
            comment_words = jieba.lcut(comment)

            if book =='':
                continue

            #将评论添加到对应的书籍中
            #book_conments.get(book, [])方法可以获取book对应的评论列表，如果book不存在，则返回空列表
            book_conments[book] = book_conments.get(book, []) 
            book_conments[book].extend(comment_words)
    return book_conments

if __name__ == "__main__":

    #加载停用词
    #line.strip()方法可以去除文本中的空格
    stop_words = [line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8')]

    #加载图书评论
    book_conments = load_data('doubanbook_top250_comments_fixed.txt')
    # print(book_conments)
    print(len(book_conments))

    #提取书名和评论文本
    book_names = []
    book_comms = []
    for book, comments in book_conments.items():
        book_names.append(book)
        book_comms.append(''.join(comments))


    #构建TF-IDF特征矩阵
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf_matrix = vectorizer.fit_transform(book_comms)

    #计算相似度
    similarity_matrix = cosine_similarity(tfidf_matrix)

    #推荐
    book_list = list(book_conments.keys())
    print(book_list)
    book = input('请输入书名：')
    book_index = book_list.index(book)

    #找到相似的书籍
    recommend_book_index = np.argsort(-similarity_matrix[book_index])[1:11]

    #输出推荐的书籍
    for idx in recommend_book_index:
        print(f'《{book_names[idx]}》 \t 相似度：{similarity_matrix[book_index][idx]:.4f}')
    print()


