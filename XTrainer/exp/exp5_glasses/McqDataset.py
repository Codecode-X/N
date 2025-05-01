import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features


ANSWERTYPE_ID_MAP = {
    'hybrid': 0,
    'negative': 1,
    'positive': 2
}

class McqDataset(Dataset):
    """
    Multiple Choice Question dataset
    """
    def __init__(self, csv_path, transform=None, device='cuda'):
        """
        Args:
            csv_path: Path to the CSV file with MCQ data
            clip_model: CLIP model for feature extraction
            lens_model: CLIPGlassesLens model for feature extraction
            transform: Optional transform to be applied on images
        """
        self.device = device
        self.csv_path = csv_path
        try:
            self.data = pd.read_csv(csv_path, encoding='gbk')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        self.transform = transform
        
        # Preprocess all data
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - Load from cache if available
            - Otherwise extract features and save to cache
        """
        # Create cache file path based on CSV path
        csv_hash = hashlib.md5(self.data.to_json().encode()).hexdigest()[:10]
        cache_path = f"MCQ_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"正在加载MCQ数据集 cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.image_features = cached_data['image_features']
            self.captions_feats = cached_data['captions_feats']
            self.level_H = cached_data['level_H_list']
            self.labels = cached_data['labels']
            self.types = cached_data['types']
            # self.types = [ANSWERTYPE_ID_MAP.get(d, -1) for d in cached_data['types']] # 改为在存缓存时转换
            print(f"Loaded {len(self.image_features)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        
        self.image_features = []
        self.captions_feats = [] 
        self.level_H = []
        self.labels = [] # Correct answer index (0-3)
        self.types = [] # 正确答案的类型 (0-2)
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            img_path = row['image_path']

            # 提取图像特征
            img_feature = extract_img_features(img_path) # 原始CLIP提取的图像特征
            self.image_features.append(img_feature)
            
            # Extract text features for all options
            row_h = []
            row_level_h = []

            for i in range(4):
                caption = row[f'caption_{i}']
                h, level_h_list = extract_sentence_features(caption) # 原始CLIP提取的文本特征 | h:[embed_dim], level_h_list:[embed_dim]*num_layers
                row_h.append(h) 
                row_level_h.append(level_h_list)
            self.captions_feats.append(row_h)
            self.level_H.append(row_level_h)
            self.labels.append(row['correct_answer'])
            # self.types.append(row['correct_answer_template'])
            self.types.append(ANSWERTYPE_ID_MAP.get(row['correct_answer_template'], -1))
        
        print(f"Saving features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'captions_feats': self.captions_feats,
            'level_H_list': self.level_H,
            'labels': self.labels,
            'types': self.types
        }, cache_path)
        print(f"Cached {len(self.image_features)} samples.")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            img_feature: Image features [embed_dim]
            caption_feat: List of positive features for each option [4, embed_dim]
            level_h_list: List of negative features for each option [4, embed_dim]
            label: Correct answer index (0-3)
            answer_type: Correct answer type (0-2) | hybrid:0, negative:1, positive:2
        """
        img_feature = torch.tensor(self.image_features[idx], dtype=torch.float32)
        caption_feat = torch.from_numpy(np.array(self.captions_feats[idx])).float() # [num_captions=4, embed_dim]
        level_h_list = torch.from_numpy(np.array(self.level_H[idx])).float() # [num_captions=4, num_layers, embed_dim]
        label = torch.tensor(self.labels[idx], dtype=torch.long) # [1]
        answer_type = torch.tensor(self.types[idx], dtype=torch.long) # [1]
        
        return img_feature, caption_feat, level_h_list, label, answer_type
    
    
def evaluate_model_mcq(model, data_loader, test_raw_clip=False, device='cuda'):
    """
    Evaluate the CLIPGlassesFrame model on the validation set.
    
    参数:
        - model: The CLIPGlassesFrame model
        - data_loader: DataLoader for the validation set
        - test_raw_clip: Whether to test the raw CLIP model
        - device: Device to run the model on (CPU or GPU)
        
    返回:
        - val_acc: Validation accuracy
        - val_loss: Validation loss
        - all_predictions: List of all predictions
        - all_labels: List of all true labels
    """
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img_features, caption_feats, level_H_list, labels, answer_types in tqdm(data_loader, desc="Extract feats"):
            """
                img_feature: Image features [embed_dim]
                caption_feat: List of positive features for each option [4, embed_dim]
                level_h_list: List of negative features for each option [4, embed_dim]
                label: Correct answer index (0-3)
                answer_type: Correct answer type | hybrid,negative,positive
            """            
            img_features = img_features.to(device) # [batch_size, embed_dim]
            caption_feats = caption_feats.to(device) # [batch_size, num_options, embed_dim]
            level_H_list = level_H_list.to(device) # [batch_size, num_options, num_layers, embed_dim]
            labels = labels.to(device) # [batch_size]
            answer_types = answer_types.to(device) # [batch_size]
            
            # Get scores for all options
            num_options = caption_feats.shape[1]
            # Process each option
            all_scores = [None] * num_options  # [batch_size, batch_size] * num_options
            for i in range(num_options):
                # mini-batch中每个图像和第i个选项文本特征的匹配得分
                choice_h = caption_feats[:, i] # [batch_size, embed_dim]
                level_h_list = level_H_list[:, i] # [batch_size, num_layers, embed_dim]
                scores, _ = model(img_features, choice_h, level_h_list) # [batch_size, batch_size]
                all_scores[i] = scores # [batch_size, batch_size]
            
            # # 错误方式（直接使用全矩阵）
            # logits = torch.stack(all_scores, dim=1)  # 得到 [B, C, B]
            # 正确方式（取对角线）
            logits = torch.stack([s.diag() for s in all_scores], dim=1) # 只计算和图片对应的选项得分 [B, C]
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, labels)
            
            val_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Print metrics
    val_acc = 100. * val_correct / val_total
    print(f"predicted: {all_predictions}")
    print(f"labels: {all_labels}")
    print("="*50)
    print(f"Validation Accuracy: {val_acc:.2f}%")
    return val_acc, val_loss, all_predictions, all_labels