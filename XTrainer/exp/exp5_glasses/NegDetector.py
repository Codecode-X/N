import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import extract_all_sentence_features
from tqdm import tqdm


class NegationDetector(nn.Module):
    """
    轻量级否定分类器
    输入: CLIP文本特征 [batch_size, embed_dim]
    输出: 二分类概率 [batch_size, 1]
    """
    def __init__(self, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, h):
        """
        参数:
            - h: CLIP文本特征 [batch_size, embed_dim]
        返回:
            - logits: 二分类概率 [batch_size, 1] | 1: 否定样本, 0: 肯定样本
        """
        return torch.sigmoid(self.classifier(h))
    
    @staticmethod        
    def load_model(cfg):
        model = NegationDetector()
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"正在加载 NegationDetector 模型权重: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=True))
        model = model.to(cfg['device'])
        model.eval()
        return model

class NegationDataset(Dataset):
    def __init__(self, csv_path, batch_size=64, device='cuda'):
        """
        csv_path: 包含'positive'和'negative'两列的CSV文件
        """
        
        # 创建缓存文件路径
        cache_path = os.path.basename(csv_path).split('.')[0] + "_features_cache.pt"
        
        # 检查缓存是否存在
        if os.path.exists(cache_path): 
            print(f"Loading features from cache: {cache_path}")
            cached_data = torch.load(cache_path, weights_only=False)
            self.features = cached_data['features']
            self.labels = cached_data['labels']
        else:
            try:
                df = pd.read_csv(csv_path, encoding='utf-8', engine='python', on_bad_lines=lambda line: print(f"跳过行: {line}"))
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='gbk', engine='python', on_bad_lines=lambda line: print(f"跳过行: {line}"))
            self.texts = []
            self.labels = []
            
            # 处理肯定样本
            positive_texts = df['positive'].dropna().tolist()
            self.texts.extend(positive_texts)
            self.labels.extend([0] * len(positive_texts))  # 0: 肯定样本

            # 处理否定样本
            negative_texts = df['negative'].dropna().tolist()
            self.texts.extend(negative_texts)
            self.labels.extend([1] * len(negative_texts))  # 1: 否定样本
            
            print(f"Extracting features and creating cache: {cache_path}...")
            
            # 分批次处理特征提取
            self.features = []
            with torch.no_grad():
                for i in tqdm(range(0, len(self.texts), batch_size), desc="Processing data"):
                    batch_texts = self.texts[i:i + batch_size]
                    # 提取当前批次的特征
                    batch_features = extract_all_sentence_features(batch_texts)
                    self.features.append(torch.from_numpy(batch_features))

            # 合并所有批次的特征
            self.features = torch.cat(self.features, dim=0)

            # 保存到缓存
            print(f"Saving features to cache: {cache_path}")
            torch.save({'features': self.features, 'labels': torch.tensor(self.labels, dtype=torch.long)}, cache_path)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx].clone().detach().float()

def train_detector(csv_path, save_path="best_NegDet.pth", epochs=20, lr=1e-4):
    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = NegationDetector().to(device)
    dataset = NegationDataset(csv_path, batch_size=64, device=device)
    save_path = os.path.join(current_dir, save_path)
    
    # 划分训练验证集
    train_data, val_data = train_test_split(dataset, test_size=0.2, stratify=dataset.labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128)
    
    # 优化器
    optimizer = torch.optim.AdamW(detector.parameters(), lr=lr, weight_decay=1e-5)
    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    
    # 训练循环
    best_acc = 0.0
    best_recall = 0.0  # 保存最佳召回率
    best_epoch = 0
    for epoch in range(epochs):
        detector.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = detector(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        # 验证
        detector.eval()
        correct = 0
        total = 0
        true_positives = 0
        false_negatives = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                outputs = detector(features).squeeze()
                preds = (outputs > 0.5).float().cpu()
                correct += (preds == labels.cpu()).sum().item()
                total += labels.size(0)
                # 计算召回率
                true_positives += ((preds == 1) & (labels == 1)).sum().item() # 预测为1且标签为1的样本数
                false_negatives += ((preds == 0) & (labels == 1)).sum().item() # 预测为0但标签为1的样本数
        
        # acc
        val_acc = correct / total
        val_recall = true_positives / (true_positives + false_negatives)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_data):.4f}, Val Acc={val_acc:.4f}, Val Recall={val_recall:.4f}")
        scheduler.step()
        
        if val_recall > best_recall:
            best_recall = val_recall
            best_acc = val_acc
            best_epoch = epoch + 1
            print(f"Saving best model to {save_path} with recall={best_recall:.4f} and acc={best_acc:.4f}")
            torch.save(detector.state_dict(), save_path)
    
    print(f"Best model: recall={best_recall:.4f}, acc={best_acc:.4f} at epoch {best_epoch}")
    return detector

def predict_negation(detector, texts):
    """
    参数:
        - detector: 训练好的模型
        - texts: 待检测文本列表
    返回:
        - preds: 否定检测结果列表
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector.eval()
    
    with torch.no_grad():
        features = extract_all_sentence_features(texts)
        features = torch.from_numpy(features).to(device, dtype=torch.float32)
        outputs = detector(features).squeeze()
        logits = outputs.float().cpu().numpy()
        preds = (outputs > 0.5).float().cpu().numpy()
        
    
    return logits, preds

if __name__ == "__main__":
    # 训练
    # csv_path = "/root/NP-CLIP/NegBench/data/PN.csv"  # 替换为你的CSV文件路径
    # detector = train_detector(csv_path, epochs=100, lr=1e-3)
    # torch.save(detector.state_dict(), os.path.join(current_dir, "last_NegDet.pth"))
    
    # 测试
    detector = NegationDetector.load_model({
        'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth', 
        'device': 'cuda'})
    texts = [
        "a photo of a dog", 
        "a photo of a chair",
        "a photo of a human",
        "a photo of a woman",
        "there is a dog",
        '???',
        "there is no dog",
        "a photo of a none",
        "a photo of a nonesense object",
        "a photo of a no object",
        "a photo of a no object",
        "None",
        "a woman without glasses",
    ]
    logits, preds = predict_negation(detector, texts)
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    print(logits)
    print(preds)
