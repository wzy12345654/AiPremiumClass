import pandas as pd

def load_and_select_columns(file_path, columns_to_keep, output_file_path):
    """
    加载CSV文件并保留指定的列，并将处理后的数据保存为新的CSV文件。

    参数:
    - file_path: str, CSV文件的路径。
    - columns_to_keep: list, 需要保留的列名列表。
    - output_file_path: str, 新文件保存路径。

    返回:
    - pandas DataFrame, 只包含指定列的数据。
    """
    # 加载数据
    data = pd.read_csv(file_path)

    # 打印列名，确认数据加载正确
    print("数据加载成功！列名如下：")
    print(data.columns)

    # 保留指定的列
    selected_data = data[columns_to_keep]

    print(f"已保留以下列：{columns_to_keep}")

    # 保存处理后的数据到新 CSV 文件
    selected_data.to_csv(output_file_path, index=False)
    print(f"处理后的数据已保存到: {output_file_path}")

    return selected_data

# 测试函数
file_path = '/mnt/data_1/zfy/4/week6/资料/homework_2/Summary of Weather.csv'  # 替换为你的实际文件路径
columns_to_keep = ['MaxTemp','MinTemp','MeanTemp','YR','MO','DA']  # 需要保留的列
output_file_path = '/mnt/data_1/zfy/4/week6/资料/homework_2/processed_weather_data.csv'  # 新文件保存路径

data = load_and_select_columns(file_path, columns_to_keep, output_file_path)

# 查看处理后的数据
print(data.head())
