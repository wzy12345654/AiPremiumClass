""
# 数据清洗
# 1.只保留评分和评价内容两列，并修改列名
# 2.删除评论内容为“此用户未填写评价内容”，“您没有填写内容，默认好评”的记录
# 3.删除评论内容都为标点符号，和内容为空的记录
# 4.去掉长度超过128的记录
# 5.将 5 分制评分转换为 0（差评）和 1（好评）
""
# 导入必要的库
import pandas as pd
try:
    import regex as re  # 优先使用 regex 模块以支持 \p 语法
except ImportError:
    import re

# 读取文件
file_path = 'data/jd_comment_data.xlsx'
try:
    df = pd.read_excel(file_path)
    print("文件读取成功")
    print("文件列名:", df.columns)  # 打印列名用于调试
except FileNotFoundError:
    print(f"错误：未找到文件 {file_path}，请检查文件路径。")
except Exception as e:
    print(f"读取文件时出错：{e}")
else:
    # 只保留评分和评价内容两列，并修改列名
    try:
        df = df[['评分（总分5分）(score)', '评价内容(content)']]
        df.columns = ['5分制评分', '评价内容']
    except KeyError as e:
        print(f"错误：DataFrame 中不存在列 {e}，请检查列名。")
    else:
        # 删除评论内容为“此用户未填写评价内容”的记录
        df = df[df['评价内容'] != '此用户未填写评价内容']
        # 删除评论内容为 您没有填写内容，默认好评 的记录
        df = df[df['评价内容']!= '您没有填写内容，默认好评']
        # 删除评论内容为空的记录
        df = df.dropna(subset=['评价内容'])
        # 删除评论内容都为标点符号的记录
        pattern = re.compile(r'^[\p{P}\p{S}]+$', re.UNICODE)
        df = df[~df['评价内容'].apply(lambda x: bool(pattern.fullmatch(str(x))))]
        # 去掉长度超过128的记录
        df = df[df['评价内容'].apply(len) <= 128]

        # 将 5 分制评分转换为 0（差评）和 1（好评）
        def convert_rating(rating):
            if 1 <= rating <= 3:
                return 0
            elif 4 <= rating <= 5:
                return 1
            else:
                return None

        df['评价类型'] = df['5分制评分'].apply(convert_rating)
        # 删除原始的 5 分制评分列
        df = df.drop(columns=['5分制评分'])

        # 打印评价内容的最大长度
        max_length = df['评价内容'].apply(len).max()
        print(f"评价内容的最大长度为：{max_length}")

        # 保存为 xlsx 文件
        xls_path = 'data/jd_comment_result.xlsx'
        df.to_excel(xls_path, index=False)

        # 选出100条数据，其中50条评价为好评，50条评价为差评
        df = df.sample(frac=1, random_state=42)  # 打乱数据
        df = df.groupby('评价类型').head(50)  # 取出前50条
        df.to_excel('data/jd_comment_result_100.xlsx', index=False)

        print(f'数据清洗完成，已保存至 {xls_path}')

