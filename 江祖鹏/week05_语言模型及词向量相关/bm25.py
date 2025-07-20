import csv
import jieba
import numpy as np


def load_data(filename, stop_words):
    """加载图书评论数据并过滤停用词"""
    book_comments = {}
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for item in reader:
            book = item['book']
            comment = item['body']
            if not book:  # 过滤空书名
                continue
            
            # 分词并过滤停用词
            words = [word for word in jieba.lcut(comment) if word not in stop_words]
            
            # 按书籍聚合评论
            if book not in book_comments:
                book_comments[book] = []
            book_comments[book].extend(words)
    return book_comments


def bm25(book_docs, k=1.5, b=0.75):
    """优化后的 BM25 算法实现"""
    # 文档统计初始化
    N = len(book_docs)
    doc_lengths = [len(doc) for doc in book_docs]
    avg_doc_len = np.mean(doc_lengths)
    
    # 词频统计
    word_doc_freq = {}  # 包含词的文档数
    doc_term_dict = [{} for _ in range(N)]  # 每本书的词频
    
    for i, doc in enumerate(book_docs):
        unique_words = set()
        for word in doc:
            doc_term_dict[i][word] = doc_term_dict[i].get(word, 0) + 1
            unique_words.add(word)
        for word in unique_words:
            word_doc_freq[word] = word_doc_freq.get(word, 0) + 1
    
    # 计算 IDF
    vocabulary = list(word_doc_freq.keys())
    idf = {word: np.log((N - word_doc_freq[word] + 0.5) / (word_doc_freq[word] + 0.5)) 
           for word in vocabulary}
    
    # 预转换为集合加速查找
    book_docs_sets = [set(doc) for doc in book_docs]
    
    # 计算相似度矩阵
    similarity_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                similarity_matrix[i][j] = 1.0
                continue
            
            score = 0
            for word in book_docs_sets[i]:
                if word in book_docs_sets[j]:
                    tf = doc_term_dict[j][word]
                    idf_val = idf[word]
                    numerator = tf * (k + 1)
                    denominator = tf + k * (1 - b + b * doc_lengths[j]/avg_doc_len)
                    score += idf_val * (numerator / denominator)
            
            similarity_matrix[i][j] = score
    
    return similarity_matrix


if __name__ == "__main__":
    # 加载停用词
    stop_words = set(line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8'))
    
    # 加载数据（传递停用词参数）
    book_comments = load_data('doubanbook_top250_comments_fixed.txt', stop_words)
    book_names = list(book_comments.keys())
    book_docs = list(book_comments.values())  # 每本书的分词列表
    
    # 计算相似度
    print("正在计算 BM25 相似度矩阵...")
    similarity_matrix = bm25(book_docs)
    
    # 推荐逻辑
    print("可用书籍列表：", book_names)
    try:
        target_book = input("请输入书名：")
        book_index = book_names.index(target_book)
        
        # 获取相似书籍索引
        sorted_indices = np.argsort(-similarity_matrix[book_index])[1:11]
        
        # 输出结果
        print(f"\n与《{target_book}》相似的书籍：")
        for idx in sorted_indices:
            print(f"《{book_names[idx]}》 相似度：{similarity_matrix[book_index][idx]:.4f}")
    
    except ValueError:
        print("错误：输入的书名不存在！")