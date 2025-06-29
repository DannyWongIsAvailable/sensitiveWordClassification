import pandas as pd


# 读取CSV文件
def read_and_extract_categories(csv_file):
    # 加载CSV文件
    data = pd.read_csv(csv_file)

    # 检查"类别"字段是否存在
    if '类别' not in data.columns:
        raise ValueError("CSV文件中没有'类别'字段")

    # 提取所有唯一类别
    categories = data['类别'].unique()

    # 输出所有类别
    print("所有唯一类别：")
    print(categories)


# 主程序
if __name__ == "__main__":
    # 指定CSV文件路径
    csv_file = 'dataset/train.csv'  # 请替换为你的实际文件路径
    try:
        read_and_extract_categories(csv_file)
    except Exception as e:
        print(f"发生错误: {e}")
