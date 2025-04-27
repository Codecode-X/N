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


class McqDataset(Dataset):
    """Multiple Choice Question dataset"""
    def __init__(self, csv_path, clip_model, lens_model, transform=None):
        """
        Args:
            csv_path: Path to the CSV file with MCQ data
            clip_model: CLIP model for feature extraction
            lens_model: CLIPGlassesLens model for feature extraction
            transform: Optional transform to be applied on images
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.clip_model = clip_model
        self.lens_model = lens_model
        self.transform = transform
        self.device = next(lens_model.parameters()).device
        
        # Preprocess all data
        self._preprocess_features()
        
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - 如果有预处理数据文件，则直接加载
            - 如果没有则提取，并保存预处理数据文件，下次直接加载
        """
        # Create cache file path based on CSV path
        csv_hash = hashlib.md5(self.data.to_json().encode()).hexdigest()[:10]
        
        dataset_name = "COCO" if "COCO" in self.csv_path else "VOC2007"
        cache_path = f"{dataset_name}_mcq_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path)
            with torch.no_grad():
                self.image_features = cached_data['image_features']
                self.pos_features = cached_data['pos_features']
                self.neg_features = cached_data['neg_features']
                self.labels = cached_data['labels']
            print(f"Loaded {len(self.labels)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        self.image_features = []
        self.pos_features = []  # List of lists (4 options per sample)
        self.neg_features = []  # List of lists (4 options per sample)
        self.labels = []
        
        for idx, row in tqdm.tqdm(self.data.iterrows(), total=len(self.data)):
            # Extract image features
            img_path = row['image_path']
            img_features = extract_img_features(img_path)
            
            # Extract text features for all options
            option_pos_features = []
            option_neg_features = []
            
            for i in range(4):
                caption = row[f'caption_{i}']
                text_features = extract_sentence_features(caption)
                text_features_tensor = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get positive and negative features using lens model
                with torch.no_grad():
                    h_pos, h_neg = self.lens_model(text_features_tensor)
                
                option_pos_features.append(h_pos.cpu().numpy()[0])
                option_neg_features.append(h_neg.cpu().numpy()[0])
            
            self.image_features.append(img_features)
            self.pos_features.append(option_pos_features)
            self.neg_features.append(option_neg_features)
            self.labels.append(row['correct_answer'])
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'pos_features': self.pos_features,
            'neg_features': self.neg_features,
            'labels': self.labels
        }, cache_path)
        print(f"Cached {len(self.labels)} samples")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_features: Image features [embed_dim]
            pos_features: List of positive features for each option [4, embed_dim]
            neg_features: List of negative features for each option [4, embed_dim]
            label: Correct answer index (0-3)
        """
        img_features = torch.tensor(self.image_features[idx], dtype=torch.float32)
        
        # Stack all option features
        pos_features = torch.tensor(np.stack(self.pos_features[idx]), dtype=torch.float32)
        neg_features = torch.tensor(np.stack(self.neg_features[idx]), dtype=torch.float32)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return img_features, pos_features, neg_features, label
    
    

def evaluate_model_mcq(model, data_loader, device):
    """
    Evaluate the CLIPGlassesFrame model on the validation set.
    
    参数:
        - model: The CLIPGlassesFrame model
        - data_loader: DataLoader for the validation set
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
        for img_features, text_features, _,  labels in tqdm.tqdm(data_loader, desc="Validation"):
            img_features = img_features.to(device) # [batch_size, embed_dim]
            text_features = text_features.to(device) # [batch_size, num_options, embed_dim]
            labels = labels.to(device)
            
            # Get scores for all options
            num_options = text_features.shape[1]
            # Process each option
            all_scores = [None] * num_options  # [batch_size, batch_size] * num_options
            for i in range(num_options):
                # mini-batch中每个图像和第i个选项文本特征的匹配得分
                choice_features = text_features[:, i] # [batch_size, embed_dim]
                scores, _ = model(img_features, choice_features) # [batch_size, batch_size]
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
    return val_acc, val_loss, all_predictions, all_labels