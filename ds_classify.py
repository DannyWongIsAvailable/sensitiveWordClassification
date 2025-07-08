import json

import pandas as pd
from openai import OpenAI, BadRequestError
import time

# 初始化API客户端
client = OpenAI(
    base_url="https://api.deepseek.com/",  # API地址
    api_key="sk-e87721e3a0974f478b9362a55c1d2b71"  # API密钥
)

# 读取CSV文件
df = pd.read_csv('dataset/test.csv')

# 创建一个空的列表用于存储分类结果
categories = []

# 定义循环逐行分类的函数
def classify_text(text):
    try:
        # 构造API请求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": """#### 定位
                    - 智能助手名称 ：违禁信息鉴定分类专家
                    - 主要任务 ：对输入的违禁信息进行自动分类，识别其所属的违禁信息种类。
                    - 能力：
                      - 文本分析 ：能够准确分析违禁信息文本的内容和结构。
                      - 分类识别 ：根据分析结果，将违禁信息文本分类到预定义的种类中。
                    - 知识储备：
                      - 违禁信息种类：
                        - 种族歧视
                        - 政治敏感
                        - 微侵犯(MA)
                        - 色情
                        - 犯罪
                        - 地域歧视
                        - 基于文化背景的刻板印象(SCB)
                        - 宗教迷信
                        - 性侵犯(SO)
                        - 基于外表的刻板印象(SA)
                    - 使用说明：
                      - 输入 ：一段违禁信息文本。
                      - 输出 ：只输出违禁信息文本所属的种类，不需要额外解释。
                    """
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        # 获取分类结果
        category = response.choices[0].message.content.strip()
        return category
    except BadRequestError as e:
        # 如果捕获到BadRequestError，返回'政治敏感'，并继续
        print(f"BadRequestError 错误：{e}")
        return '政治敏感'
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        return '解析错误'

    except Exception as e:
        print(f"未知错误: {type(e).__name__}: {e}")
        return '处理失败'

# 循环遍历每一行，进行分类
for index, row in df.iterrows():
    try:
        text = row['文本']
    except KeyError:
        print(f"行{index}缺少文本字段")
        categories.append('缺失文本')  # 添加默认值
        continue

    # 调用分类函数
    category = classify_text(text)

    # 将分类结果添加到列表中
    categories.append(category)

    print(f"正在处理第{row['id']}行: {text}，类别为：{category}")
    # 为了避免触发API限制，适当休眠一下
    time.sleep(1)

# 创建完整版本DataFrame（包含id,文本,类别）
full_df = pd.DataFrame({
    'id': df['id'],
    '文本': df['文本'],
    '类别': categories
})

# 创建精简版本DataFrame（只包含id,类别）
simple_df = pd.DataFrame({
    'id': df['id'],
    '类别': categories
})

# 保存两个CSV文件
full_df.to_csv('ds_cls_full.csv', index=False)
simple_df.to_csv('ds_cls_simple.csv', index=False)

print("分类完成，已保存两个文件：")
print("- ds_cls_full.csv (包含id,文本,类别)")
print("- ds_cls_simple.csv (只包含id,类别)")
