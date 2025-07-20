# excel2csv_single.py

import pandas as pd

# === 输入输出路径 ===
excel_path = "/mnt/data_1/zfy/self/homework/10/jd_comment_data.xlsx"
csv_path = "/mnt/data_1/zfy/self/homework/10/jd_comment_data.csv"

def convert_first_sheet_to_csv(excel_path, csv_path):
    # 读取 Excel 的第一个工作表
    df = pd.read_excel(excel_path, sheet_name=0)
    
    # 保存为 CSV 文件（不带索引）
    df.to_csv(csv_path, index=False)
    print(f"转换完成，CSV 文件保存至：{csv_path}")

if __name__ == "__main__":
    convert_first_sheet_to_csv(excel_path, csv_path)
