import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

categories = ['种族歧视', '政治敏感', '微侵犯(MA)', '色情', '犯罪', '地域歧视', '基于文化背景的刻板印象(SCB)', '宗教迷信',
              '性侵犯(SO)', '基于外表的刻板印象(SA)']

class RoBertaDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, is_train=True):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.label_map = {category: idx for idx, category in enumerate(categories)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['文本']
        if not isinstance(text, str):
            text = str(text)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        if self.is_train:
            category = self.data.iloc[idx]['类别']
            if category not in self.label_map:
                raise KeyError(f"类别 '{category}' 不在预定义的 categories 列表中，请检查数据集中的类别拼写。")
            label = self.label_map[category]
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            }


def create_dataloader(file_path, tokenizer, max_length, batch_size, is_train=True):
    dataset = RoBertaDataset(file_path, tokenizer, max_length, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
