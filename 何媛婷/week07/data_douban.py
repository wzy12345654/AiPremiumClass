import pandas as pd
import re
import jieba


# 加载数据集
def load_data():
    """
    加载数据 从网址下载的CSV文件
    """
    data_path = './data/DMSC.csv'  # 替换为实际路径
    df = pd.read_csv(data_path) 
    return df



# 根据作业要求，新增一列 label 根据 star字段处理
def fetch_star2cate(star):
    """
    评论得分1-2	表示negative  取值 1
    评论得分4-5 代表positive  取值 0
    """
    if 1 <= star <= 2:
        return 1
    elif 4 <= star <= 5:
        return 0
    else:
        return None

# 去除特殊符号和数字
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)  
    return text

# 分词处理
def fetch_text_wcut(str):
    words = jieba.lcut(clean_text(str))
    all_cuts = [params for word in words for params in word]
    return all_cuts