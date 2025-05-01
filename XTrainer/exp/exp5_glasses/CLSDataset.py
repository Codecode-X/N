import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
import torch
import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import hashlib
from get_model import extract_img_features, extract_sentence_features, Clip_model



class CLSDataset(Dataset):
    """
    ImageNet CLS dataset
    
    /root/NP-CLIP/NegBench/data/imagenet_val.csv:
    id,image_path,label
    0,val/ILSVRC2012_val_00000001.JPEG,tench
    1,val/ILSVRC2012_val_00000002.JPEG,goldfish
    2,val/ILSVRC2012_val_00000003.JPEG,great_white_shark
    """
    def __init__(self, csv_path, clip_model, lens_model, transform=None):
        """
        Args:
            csv_path: Path to the ImageNet dataset csv file
            clip_model: CLIP model for feature extraction
            lens_model: CLIPGlassesLens model for feature extraction
            transform: Optional transform to be applied on images
        """
        self.csv_path = Path(csv_path)
        self.clip_model = clip_model
        self.lens_model = lens_model
        self.transform = transform
        self.device = next(lens_model.parameters()).device

        # Load CSV data
        self.df = pd.read_csv(self.csv_path)
        
        # 从中随机选5000个样本
        if len(self.df) > 5000:
            self.df = self.df.sample(n=5000, random_state=3407).reset_index(drop=True)

        # Extract unique classes and create a mapping
        self.class_list = self.df['label'].unique().tolist()
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.class_list)}

        # Preprocess and cache features
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - If preprocessed data file exists, load it directly
            - Otherwise extract features and save preprocessed data for future use
        """
        # Create cache file path based on CSV path
        csv_hash = hashlib.md5(self.df.to_json().encode()).hexdigest()[:10]
        cache_path = f"imagenet_cls_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path)
            with torch.no_grad():
                self.image_features = cached_data['image_features']
                self.labels = cached_data['labels']
                self.text_features = cached_data['text_features']
                self.level_H = cached_data['level_H_list']
            print(f"Loaded {len(self.labels)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        self.image_features = []
        self.text_features = []
        self.level_H = [] # 原始CLIP每一层输出的图像特征列表
        self.labels = []
        
        # Extract image features for all images
        for idx, row in tqdm.tqdm(self.df.iterrows(), total=len(self.df)):
            # Extract image features
            img_path = str(row['image_path'])
            img_features = extract_img_features(img_path)
            # Store image features and label
            self.image_features.append(img_features)
            self.labels.append(self.class_to_id[row['label']])
        
        # Extract text features for all classes
        print("Extracting text features for all classes...")
        for class_name in tqdm.tqdm(self.class_list):
            # Create prompt for class
            prompt = f"A photo of a {class_name.replace('_', ' ')}."
            # Extract text features
            h, level_h  = extract_sentence_features(prompt)
            self.text_features.append(h)
            self.level_H.append(level_h)
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'labels': self.labels,
            'text_features': self.text_features, # [num_classes, embed_dim]
            'level_H_list': self.level_H # [num_classes, embed_dim]
        }, cache_path)
        print(f"Cached {len(self.labels)} samples with {len(self.class_list)} classes")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns:
            image_features: Image features [embed_dim]
            text_features: Text features [num_classes, embed_dim]
            level_H: Level H features [num_classes, num_layers, embed_dim]
            label: Class label (long)
        """
        img_features = torch.tensor(self.image_features[idx], dtype=torch.float32) # [embed_dim]
        text_features = torch.tensor(self.text_features, dtype=torch.float32) # [num_classes, embed_dim]
        level_H = torch.tensor(self.level_H, dtype=torch.float32) # [num_classes, num_layers, embed_dim]
        label = torch.tensor(self.labels[idx], dtype=torch.long) # long
        return img_features, text_features, level_H, label


def evaluate_model_CLS(model, data_loader, test_raw_clip=False, device='cuda'):
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
    dtype = model.dtype
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img_features, all_class_feats, level_H, labels in tqdm(data_loader, desc="Extract feats"):
            """
                img_features: Image features [batch_size, embed_dim]
                all_class_feats: Text features [batch_size, num_options, embed_dim]
                level_H: Level H features [batch_size, num_options, num_layers, embed_dim]
                labels: Class labels [batch_size]
            """            
            img_features = img_features.to(device, dtype=dtype) # [batch_size, embed_dim]
            all_class_feats = all_class_feats.to(device, dtype=dtype) # [batch_size, num_options, embed_dim]
            level_H = level_H.to(device, dtype=dtype) # [batch_size, num_options, num_layers, embed_dim]
            labels = labels.to(device, dtype=dtype) # [batch_size]
            answer_types = answer_types.to(device, dtype=dtype) # [batch_size]
            
            # Get scores for all options
            num_cls = all_class_feats.shape[1]
            # Process each option
            all_scores = [None] * num_cls  # [batch_size, batch_size] * num_options
            
            for i in range(num_cls):
                # mini-batch中每个图像和第i个选项文本特征的匹配得分
                choice_h = all_class_feats[:, i] # [batch_size, embed_dim]
                level_h_list = level_H[:, i] # [batch_size, num_layers, embed_dim]
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
 