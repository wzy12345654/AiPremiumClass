import csv #专门用于结构化数据的读取
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi 
import numpy as np
#数据预处理，将数据加工成字典类型，一个书名对应所有的评论，评论要是分词器分好后的，如下
# book_comments = {
#     "《百年孤独》": ['这是', '一本', '经典', ...],
#     "《小王子》": ['充满', '哲理', '的小书', ...],
#     "《追风筝的人》": ['感人', '至深', '故事', ...]
# }
#数据预处理
def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f :
        #
        reader=csv.DictReader(f,delimiter='\t') #能区分标题的dictreader

        #测试一下数据是什么样的,为一组字典，每一条数据都是一个字典类型
        # i=1
        # for r in reader:
        #     print(r)
        #     if(i==5):
        #         break
        #     i=i+1

        # 图书评论信息集合
        book_comments = {}  
        for item in reader:
            #获取这一行的书名
            book = item['book']
            #获取这一行的评论
            comment = item['body']

            #使用分词器
            comment_words = jieba.lcut(comment)
            #先看看接下来的一条评论是不是也是上一本书的，如果是，就能获取到，这样就获取后再追加.
            #如果获取不到，就创建一条新的，之所以用[]，是为了防止报错，如果没获取到就创建一个
            book_comments[book] = book_comments.get(book,[])
            book_comments[book].extend(comment_words)
        return book_comments
            #print(book_comments['百年孤独'])

def comments_vectors_similarity(book_comms, method):
    # 加载停用词列表
    stop_words = [line.strip() for line in open("第五周/stopwords.txt", "r", encoding="utf-8")]
    if method == 'tfidf':
        # 构建TF-IDF特征矩阵
        vectorizer = TfidfVectorizer(stop_words=stop_words)
        matrix = vectorizer.fit_transform([' '.join(comms) for comms in book_comms])
        similarity_matrix = cosine_similarity(matrix)
    elif method == 'bm25':
        # 构建BM25模型
        tokenized_corpus = [[word for word in comms if word not in stop_words] for comms in book_comms]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # 计算每本书与其他书的相似度得分，因为bm25返回的是得分，不是矩阵，所以需要手搓矩阵
        similarity_matrix = np.zeros((len(book_comms), len(book_comms)))
        for i, query in enumerate(tokenized_corpus):
            scores = bm25.get_scores(query)
            similarity_matrix[i] = scores
        
        # 归一化相似度得分（可选）
        similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
    else:
        raise ValueError("Invalid method. Choose either 'tfidf' or 'bm25'.")
    # 计算图书之间的余弦相似度
    return similarity_matrix

if __name__ == '__main__':
    #数据预处理
    book_comments=load_data("第五周/douban_comments.txt")

    book_names = list(book_comments.keys())
    book_comms = list(book_comments.values())
    #print(book_names)

    #丢入数据训练,因为评论与书名是按照位置对应的，所以只需要丢入评论，他会计算每条评论与其他评论的相似度
    tfidf_matrix=comments_vectors_similarity(book_comms,'tfidf')

    book_name = input("请输入图书名称：")
    #找到这本书的索引
    book_idx = book_names.index(book_name)
    #放入模型，找到对应的矩阵列，排序，这里写负号说明从大到小排，从1取是因为要排除自己
    recommend_book_index = np.argsort(-tfidf_matrix[book_idx])[1:11]
    for idx in recommend_book_index:
        print(f"《{book_names[idx]}》 \t 相似度：{tfidf_matrix[book_idx][idx]:.4f}")  
    #矩阵样式如下
    # similarity_matrix = [
    #     [1.0, 0.8, 0.6],  # 《百年孤独》与其他书相似度
    #     [0.8, 1.0, 0.7],  # 《小王子》与其他书相似度
    #     [0.6, 0.7, 1.0]   # 《追风筝的人》与其他书相似度
    # ]

