
import pandas as pd
def dataProcess():
    # 读取xlsx文件
    df = pd.read_excel('/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx')

    # 提取指定字段（示例字段名为'字段1','字段2','字段3'）
    selected_columns = ['评价内容(content)', '评分（总分5分）(score)']
    df_selected = df[selected_columns]
    print(df_selected[1:10])

    # 保存为csv文件，重新指定字段的名字 content label（不包含索引）
    df_selected.columns = ['content', 'label']
    # 如果label 小于3 则为0， 大于3 则为1
    df_selected['label'] = df_selected['label'].apply(lambda x: 0 if x <= 3 else 1)
    print(df_selected[1:10])
    # 保存为csv文件，重新指定字段的名字 content label（不包含索引）
    df_selected.to_csv('/kaggle/working/jd-comments.csv', index=False)
    print("数据处理结束")