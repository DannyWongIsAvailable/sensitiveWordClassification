import pandas as pd


# 读取CSV文件并随机抽取1000条数据
def random_sample_and_reindex(csv_file, sample_size=1000):
    # 加载CSV文件
    data = pd.read_csv(csv_file)

    # 检查id列是否存在
    if 'id' not in data.columns:
        raise ValueError("CSV文件中没有'id'列")

    # 随机抽取1000条数据
    sampled_data = data.sample(n=sample_size, random_state=42)

    # 重新编排id列，按新的顺序编号
    sampled_data['id'] = range(1, len(sampled_data) + 1)

    # 输出新的数据
    print("随机抽取并重新编排id后的数据：")
    print(sampled_data.head())  # 输出前五条数据查看效果

    # 保存新的CSV文件
    sampled_data.to_csv('val.csv', index=False)


# 主程序
if __name__ == "__main__":
    # 指定CSV文件路径
    csv_file = 'dataset/val.csv'  # 请替换为实际文件路径
    try:
        random_sample_and_reindex(csv_file)
    except Exception as e:
        print(f"发生错误: {e}")
