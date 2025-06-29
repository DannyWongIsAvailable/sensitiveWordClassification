import os
import torch
from model import RoBerta  # 修改为 RoBerta 模型的导入
from data_loader import create_dataloader
from transformers import AutoTokenizer  # 修改为 AutoTokenizer
import yaml
import pandas as pd

# 读取配置文件
with open('../configs/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model']['pretrained_model_name'])

# 加载测试集
test_dataloader = create_dataloader(
    config['data']['test_path'],
    tokenizer,
    config['data']['max_seq_length'],
    config['evaluation']['eval_batch_size'],
    is_train=False
)

# 加载分类模型
num_labels = 10  # 根据分类任务中类别数设定
model = RoBerta(config['model']['pretrained_model_name'], num_labels, config['model']['dropout'])
model.load_state_dict(torch.load('../experiments/experiment_1/best_add.pt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 定义类别映射和分数映射
labels = ['种族歧视', '政治敏感', '微侵犯(MA)', '色情', '犯罪', '地域歧视', '基于文化背景的刻板印象(SCB)', '宗教迷信',
          '性侵犯(SO)', '基于外表的刻板印象(SA)']

# 评估模型
model.eval()
predicted_labels = []
predicted_scores = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 如果使用了 token_type_ids，也需要传递它们
        if 'token_type_ids' in batch:
            token_type_ids = batch['token_type_ids'].to(device)
            outputs = model(input_ids, attention_mask, token_type_ids)
        else:
            outputs = model(input_ids, attention_mask)

        # 模型输出为类别概率分布，取最大值作为预测类别
        _, predicted_label_indices = torch.max(outputs, dim=1)

        # 将预测的整数索引映射回类别名称
        predicted_labels.extend([labels[label] for label in predicted_label_indices.cpu().numpy()])

# 保存预测结果
test_data = pd.read_csv(config['data']['test_path'])

# 构建仅包含 id 和 类别 的 DataFrame
output_df = pd.DataFrame({
    'id': test_data['id'],
    '类别': predicted_labels
})
# 定义保存路径并检查文件是否存在，如果存在则递增序号
base_save_path = '../results/predictions.csv'
save_path = base_save_path

if os.path.exists(save_path):
    i = 1
    while os.path.exists(save_path):
        save_path = f"../results/predictions_{i}.csv"
        i += 1

# 保存预测结果到最终确定的路径
output_df.to_csv(save_path, encoding='utf-8', index=False)
print(f"预测结果已保存到 {save_path}")
