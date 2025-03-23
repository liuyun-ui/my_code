import os #处理文件路径
import torch #构建和训练模型
from torch.utils.data import DataLoader #用于将数据集分批次加载到模型中
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
#BertTokenizer：BERT 的分词器，用于将文本转换为模型输入格式。
#BertForSequenceClassification：BERT 模型，用于分类或回归任务。
from sklearn.model_selection import train_test_split
from tqdm import tqdm #tqdm：用于显示进度条
import  json
import  pandas as pd
import  numpy as np

model_path = os.path.expanduser(r"C:\Users\15286\Desktop\code\bert")

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# 创建 PyTorch 数据集
class BangumiDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)  # 回归任务，标签为浮点数
        }
        return item

    def __len__(self):
        return len(self.labels)

def train(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader)}")

        # 验证集评估
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.squeeze().cpu().numpy()  # 回归任务，直接取 logits
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        mse = np.mean((np.array(predictions) - np.array(true_labels)) ** 2)
        print(f"Validation MSE: {mse:.4f}")

# 主函数

def main():
    # 加载数据
    train_data = load_jsonl("comments_and_ratings.jsonl")  # 替换为你的 jsonl 文件路径
    train_texts = [item["text"] for item in train_data]
    train_labels = [item["point"] for item in train_data]
    # 划分训练集和验证集
    # train_texts, val_texts, train_labels, val_labels = train_test_split(
    #     texts, labels, test_size=0.2, random_state=42
    # )
    val_data = load_jsonl(r"C:\Users\15286\Desktop\code\catch\test.jsonl")  # 替换为你的验证集路径
    val_texts = [item["text"] for item in val_data]
    val_labels = [item["point"] for item in val_data]

    # 加载本地分词器和模型
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=1)  # 回归任务

# 数据编码
    train_encodings = tokenizer(train_texts, truncation=True, padding="max_length", max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding="max_length", max_length=128)

    # 创建数据集和 DataLoader
    train_dataset = BangumiDataset(train_encodings, train_labels)
    val_dataset = BangumiDataset(val_encodings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # 配置优化器和设备
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 开始训练
    train(model, train_loader, val_loader, optimizer, device)

    model.save_pretrained("bert_regression_model")
    tokenizer.save_pretrained("bert_regression_model")

if __name__ == "__main__":
    main()







