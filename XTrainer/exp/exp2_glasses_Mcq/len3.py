"""
    CLIP-GlassesLens: 一个轻量级模块CLIP-GlassesLens，通过处理CLIP的文本编码器的输出文本特征 h ，筛除掉其中的否定内容 h_neg，并保留下不含否定内容的肯定特征 h_pos，即 h_pos = h - h_neg。
    例如原始文本 S = "In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves." 中的肯定内容 S_pos = "In a rustic cabin, an elegant bench sits in the corner", 需要被筛掉的被否定内容 S_neg = "a camera and a gloves". 
    其中 h = Clip.encode_text(S) 为该模块的输入，l_pos = Clip.encode_text(S_pos) 为该模块的监督信号。
    你需要预测出 h_neg ，并使得 h_pos = h - h_neg 和 l_pos 尽可能相似（余弦相似度）。    
    
    输入:
        - CLIP的文本编码器的输出文本特征 h
            - 示例：原始输入句子 "In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves." 的文本特征。

    输出:
        - 肯定内容文本特征 h_pos 
            - 模型通过处理原始输入h，筛除其中的否定内容，生成肯定内容文本特征 h_pos | GT: l_pos
        - 否定内容文本特征 h_neg
            - h_neg = h - h_pos
        
    训练： 
        - 数据集1(用于生成h)：COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
        - 数据集2(用于生成h_pos)：COCO_val_retrieval.csv
        - 监督信号: 
            - 1. 提取数据集1中的caption文本，经过CLIP文本编码器的输出文本特征 h
            - 2. 提取数据集2中的caption文本，经过CLIP文本编码器的输出文本特征 l_pos
        - 通过对比学习，训练轻量级模块，使得 h_pos 和 l_pos 的相似度尽可能高，同时 h_neg 和 h_pos 的相似度尽可能低。

    数据集： 
        - 数据集1(用于生成h)：COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv:
            - 该数据集中的captions为经过重写的同时包含肯定和否定的句子，包含了否定对象和肯定对象。
            
            positive_objects,negative_objects,filepath,image_id,captions
            "['person', 'bottle', 'cup', 'knife', 'spoon', 'bowl', 'broccoli', 'carrot', 'dining table', 'oven', 'sink']","['chair', 'fork', 'car']",/root/autodl-tmp/COCO2017/val2017/000000397133.jpg,397133,"['A man in a kitchen is making pizzas, but there is no chair in sight.', 'A man in an apron stands in front of an oven with pans and bakeware nearby, without a chair in sight.', 'No fork is present, but a baker is busy working in the kitchen, rolling out dough.', 'A person stands by a stove in the kitchen, but a fork is noticeably absent.', ""At the table, pies are being crafted, while a person stands by a wall adorned with pots and pans, and noticeably, there's no fork.""]"
            ...
            "['person', 'cup', 'bowl', 'couch', 'cell phone']","['chair', 'dining table', 'bottle']",/root/autodl-tmp/COCO2017/val2017/000000015335.jpg,15335,"['No chair is visible in the image, but a group of people are sitting at a table with food.', 'No chair is visible in the image, yet a man, woman, and boy sit at a table.', 'No dining table at home; instead, a man, woman, and child are eating together at a restaurant.', 'Seated between a man and a woman is a boy; notably, there is no bottle in the image.', 'A young child, lady, and man are sitting in a booth at a table, but surprisingly, there is no chair to be seen.']"

        - 数据集2(用于生成h_pos)：COCO_val_retrieval.csv:
            - 该数据集中的captions为原始的只包含肯定对象的句子。
            
            positive_objects,negative_objects,filepath,image_id,captions
            "['person', 'bottle', 'cup', 'knife', 'spoon', 'bowl', 'broccoli', 'carrot', 'dining table', 'oven', 'sink']","['chair', 'fork', 'car']",/root/autodl-tmp/COCO2017/val2017/000000397133.jpg,397133,"['A man is in a kitchen making pizzas.', 'Man in apron standing on front of oven with pans and bakeware', 'A baker is working in the kitchen rolling dough.', 'A person standing by a stove in a kitchen.', 'A table with pies being made and a person standing near a wall with pots and pans hanging on the wall.']"
            ...
            "['person', 'cup', 'bowl', 'couch', 'cell phone']","['chair', 'dining table', 'bottle']",/root/autodl-tmp/COCO2017/val2017/000000015335.jpg,15335,"['A group of people sitting at a table with food.', 'A man, woman, and boy are sitting at a table.', 'A man, woman and child eating together at a restaurant.', 'A boy sitting between a man and a woman.', 'a young child, lady , and man sitting in a booth at a table']"

    ### 模型结构

    一个轻量级双通道门控网络，采用残差结构与稀疏约束。模型输入为CLIP（冻结）文本特征 $h \in \mathbb{R}^d$，输出否定特征 $h_{neg}$，通过动态门控机制和稀疏正则化分离语义。  

    - **门控生成层**： 

    $$ g = \sigma(W_g h + b_g) \quad (\text{门控权重}, \sigma=\text{Sigmoid}) $$  

    - **否定特征预测**： 

    $$ h_{neg} = g \odot (W_{neg} h + b_{neg}) \quad (\odot \text{为逐元素乘法}) $$  

    - **肯定特征计算**：

    $$ h_{pos} = h - h_{neg} $$  

    ---

    ### 损失函数

    #### 1. 主损失（相似度对齐）  

    强制 $h_{pos}$ 逼近监督信号 $l_{pos}$，最大化余弦相似度：

    $$ \mathcal{L}_{pos} = 1 - \frac{h_{pos} \cdot l_{pos}}{\|h_{pos}\| \|l_{pos}\|} $$  

    #### 2. 对抗损失（特征解耦）  

    最小化 $h_{pos}$ 与 $h_{neg}$ 的相似度，引入动态间隔 $m$：  

    $$ \mathcal{L}_{neg} = \max\left(0, \frac{h_{pos} \cdot h_{neg}}{\|h_{pos}\| \|h_{neg}\|} - m\right) $$  

    #### 4. 稀疏正则化  

    约束 $h_{neg}$ 的稀疏性，仅保留关键否定语义：  

    $$ \mathcal{L}_{sparse} = \|h_{neg}\|_1 $$  

    #### 总损失  

    加权融合多目标：  

    $$ \mathcal{L}_{total} = \mathcal{L}_{pos} + \lambda_1 \mathcal{L}_{neg} +  \lambda_2 \mathcal{L}_{sparse} $$  

    ---

    ### 训练策略

    #### 数据预处理  

    1. **特征预计算**：  

    - 对数据集的所有文本，用CLIP提取特征 $h$ 和 $l_{pos}$，并缓存。  

    2. **样本对齐**：  

    根据 `image_id` 将数据集1的 $h$ 与数据集2的 $l_{pos}$ 配对，确保同一场景的含否定描述与肯定描述对应。

    #### 优化设置  

    - **优化器**：AdamW（学习率 $3\times10^{-4}$，权重衰减 $0.05$）  
    - **动态间隔**：$m$ 初始为 $0.2$，随训练线性增至 $0.5$  
    - **损失权重**：$\lambda_1=0.5$, $\lambda_2=0.3$, $\lambda_3=0.1$  

    ---

    ### 训练流程  

    1. **输入**：  
    - 批次内样本 $\{h_i\}$（来自数据集1），$\{l_{pos,i}\}$（来自数据集2）  
    2. **前向传播**：  
    - 计算 $h_{neg,i} = \text{CLIP-GlassesLens}(h_i)$，$h_{pos,i} = h_i - h_{neg,i}$  
    3. **损失计算**：  
    - 依次计算 $\mathcal{L}_{pos}$、$\mathcal{L}_{neg}$、$\mathcal{L}_{sparse}$  
    4. **参数更新**：  
    - 仅更新CLIP-GlassesLens模块参数，CLIP文本编码器冻结。 
    
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import build_clip_model
from model.Clip import tokenize  # 分词器
import torch
import tqdm
import numpy as np
import pandas as pd
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB32-ep10-Caltech101-AdamW.yaml"

Clip_model = build_clip_model(config_path=config_path) # 加载CLIP模型

# 重定向输出到文件
class Logger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass
    
# 设置随机种子
def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_sentence_features(sentence:str):
    """提取单个句子的CLIP文本特征"""
    with torch.no_grad():  # 关闭梯度计算
        tokenized_text = tokenize(sentence) # [num_classes, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes, context_length]
        text_features = Clip_model.encode_text(tokenized_text) # [num_classes, embed_dim]
        return text_features.cpu().numpy()[0] # [embed_dim]


# Dataset class for training
class LensDataset(Dataset):
    def __init__(self, cfg):
        self.pos_csv_path = cfg['pos_csv_path'] # COCO_val_retrieval.csv
        self.negpos_csv_path = cfg['negpos_csv_path'] # COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
        self.data = [] # 在_preprocess_features()中填充 | [{'h': h, 'l_pos': l_pos, 'img_path': img_path}), ...]
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - 如果有预处理数据文件，则直接加载
            - 如果没有则提取，并保存预处理数据文件，下次直接加载
        """
        # Create cache file path based on CSV path
        cache_path = f"LensDataset_cache.pt"
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path}...")
            self.data = torch.load(cache_path)
            print(f"Loaded {len(self.data)} samples from cache")
            return
        
        # Read CSV files
        df_np = pd.read_csv(self.negpos_csv_path)
        df_p = pd.read_csv(self.pos_csv_path)
        
        # Create image_id lookup dictionaries
        np_by_id = {}
        p_by_id = {}
        for _, row in df_np.iterrows():
            np_by_id[row['image_id']] = row
        for _, row in df_p.iterrows():
            p_by_id[row['image_id']] = row
        
        # Match samples and extract features
        common_ids = set(np_by_id.keys()) & set(p_by_id.keys())
        
        for img_id in tqdm.tqdm(common_ids, desc="Processing data"):
            np_row = np_by_id[img_id]
            p_row = p_by_id[img_id]
            
            np_captions = eval(np_row['captions'])
            p_captions = eval(p_row['captions'])
            
            for np_cap, p_cap in zip(np_captions, p_captions):
                # print(f"Processing image_id: {img_id}, np_cap: {np_cap}, p_cap: {p_cap}")
                h = extract_sentence_features(np_cap)
                l_pos = extract_sentence_features(p_cap)
                img_path = np_row['filepath']
                self.data.append({'h': h, 'l_pos': l_pos, 'img_path': img_path})
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")  
        torch.save(self.data, cache_path)
        print(f"Preprocessed features saved to {cache_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'h': torch.tensor(self.data[idx]['h'], dtype=cfg['dtype']),
            'l_pos': torch.tensor(self.data[idx]['l_pos'], dtype=cfg['dtype']),
            'img_path': self.data[idx]['img_path']  
        }
        

class CLIPGlassesLens(nn.Module):
    """
    CLIP-GlassesLens: 一个轻量级模块CLIP-GlassesLens，
    通过处理CLIP的文本编码器的输出文本特征 h，筛除掉其中的否定内容 h_neg，并保留下不含否定内容的肯定特征 h_pos，即 h_pos = h - h_neg。
    """
    def __init__(self, cfg):
        """Initialize the CLIPGlassesLens module."""
        super(CLIPGlassesLens, self).__init__()
        
        embed_dim = Clip_model.visual.output_dim
        
        # Gate generation layer
        self.W_g = nn.Linear(embed_dim, embed_dim)
        self.b_g = nn.Parameter(torch.zeros(embed_dim, dtype=cfg['dtype']))
        
        # Negative feature prediction layer
        self.W_neg = nn.Linear(embed_dim, embed_dim)
        self.b_neg = nn.Parameter(torch.zeros(embed_dim, dtype=cfg['dtype']))
        
        # initialize W_g and W_neg weights to small values，保证初始时输出的h_pos等于h，h_neg等于0
        nn.init.zeros_(self.W_g.weight)
        nn.init.zeros_(self.W_neg.weight)
        
    def forward(self, h):
        """
        Forward pass of the CLIPGlassesLens module.
        
        Args:
            h: CLIP text features [batch_size, embed_dim]
            
        Returns:
            h_pos, h_neg: Positive and negative features
        """
        # Gate generation
        g = torch.sigmoid(self.W_g(h) + self.b_g)
        
        # Negative feature prediction
        h_neg = g * (self.W_neg(h) + self.b_neg)
        
        # Positive feature computation (residual connection)
        h_pos = h - h_neg
        
        return h_pos, h_neg


def compute_losses(cfg, h_pos, h_neg, l_pos):
    """
    Compute all loss components for CLIPGlassesLens
    
    参数:
        - cfg: 配置参数
        - h_pos: 肯定内容文本特征 [batch_size, embed_dim]
        - h_neg: 否定内容文本特征 [batch_size, embed_dim]
        - l_pos: 监督信号-肯定内容文本特征 [batch_size, embed_dim]
        - margin: 动态间隔
    
    返回:
        - total_loss: 总损失
        - loss_dict: 各个分项损失的字典 | pos2pos_sim_loss, pos2neg_sim_loss, sparse_loss
    """
    
    # Normalize features for cosine similarity
    h_pos_norm = h_pos / h_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    
    # 1. Similarity alignment loss
    pos2pos_sim = F.cosine_similarity(h_pos_norm, l_pos_norm, dim=-1) # [batch_size] 越大越好
    loss_pos2pos = 1.0 - pos2pos_sim # [batch_size] 越小越好
    
    # Total loss
    total_loss = loss_pos2pos
    
    return total_loss.mean(), {
        'pos2pos_sim_loss': loss_pos2pos.mean().item(),
    }


def train(cfg, model, device='cuda'):
    """Train the CLIPGlassesLens model"""
            
    if cfg:
        epochs = cfg['epochs']
        batch_size = cfg['batch_size']
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        train_size, val_size, test_size = cfg['split']
        num_workers = cfg['num_workers']
        early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
        
    dataset = LensDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        best_sim_loss = float('inf')
        patience_counter = 0 # Early stopping counter
        total_loss = 0
        losses = {'pos2pos_sim_loss': 0}
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            h = batch['h'].to(device)
            l_pos = batch['l_pos'].to(device)
            
            # Forward pass
            h_pos, h_neg = model(h)
            
            # Compute loss
            loss, loss_dict = compute_losses(cfg, h_pos, h_neg, l_pos)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['pos2pos_sim_loss'] += loss_dict['pos2pos_sim_loss']
        
        # Print epoch summary
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f}  pos2pos_sim_loss: {losses['pos2pos_sim_loss']/batch_count:.4f}")
        
        # Validation
        batch_sim_loss = evaluate(cfg, model, val_loader)
        
        # 早停
        if batch_sim_loss < best_sim_loss:
            best_sim_loss = batch_sim_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_clip_lens.pth'))
        else:
            print(f"💔loss improve from {best_sim_loss:.4f} to {batch_sim_loss:.4f}, cur patience_counter add to {patience_counter}")
            patience_counter += 1 # 增加耐心计数器
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        if epoch % 5 == 0:
            print(f"visualize_examples")
            visualize_examples(model, top_k=5)
        
    return model


def evaluate(cfg, model, data_loader, device='cuda'):
    """
    Evaluate the CLIPGlassesLens model
    
    参数：
        - cfg: 配置参数
        - model: CLIPGlassesLens模型
        - data_loader: 数据加载器
        - device: 设备（'cuda'或'cpu'）
        
    返回：
        - None
    """
    model.eval()
    model = model.to(device)
    pos_sim_total = 0
    neg_sim_total = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            l_pos = batch['l_pos'].to(device)
            
            # Forward pass
            h_pos, h_neg = model(h)
            
            # Normalize features for cosine similarity
            h_pos_norm = h_pos / h_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            
            # 1. Similarity alignment loss
            pos2pos_sim = F.cosine_similarity(h_pos_norm, l_pos_norm, dim=-1) # [batch_size]
            pos2neg_sim = F.cosine_similarity(h_pos_norm, h_neg_norm, dim=-1) # [batch_size]
            pos2pos_sim = torch.mean(pos2pos_sim).item()
            pos2neg_sim = torch.mean(pos2neg_sim).item()
            
            pos_sim_total += pos2pos_sim
            neg_sim_total += pos2neg_sim
    
    batch_count = len(data_loader)
    pos_sim_total = pos_sim_total / batch_count
    neg_sim_total = neg_sim_total / batch_count
    
    print(f"Evaluation results:")
    print(f"  Pos Similarity: {pos_sim_total:.4f}")
    print(f"  Neg Similarity: {neg_sim_total:.4f}")
    
    sim_loss = (1-pos_sim_total)
    return sim_loss  # 越小越好


def visualize_examples(model, top_k=5):
    examples = [
        "This image shows a person, but no motorbike is included.", # p:['person']  n:['motorbike']
        "This image features a car, but no knife is present.", # p:['car']  n:['knife']
        "A bus is included, while a car is absent.", # p:['bus']  n:['car']
        "A bird is present in this image, with no person in sight." # p:['bird']  n:['person']
    ]
    candidates = ['bench', 'camera', 'gloves', 'woman', 'screwdriver', 'egg', 'knife', 'plate', 'car', 'motorbike', 'bus', 'bird', 'person']

    """可视化模型预测效果"""
    print("="*50)
    print("CLIP-Lens Prediction Examples")
    print("="*50)
        
    # 提取候选对象文本特征
    candidate_features = []
    for obj in candidates:
        text = f"This image shows a {obj}."
        feature = extract_sentence_features(text)
        candidate_features.append(feature)
    candidate_features = torch.tensor(candidate_features, dtype=cfg['dtype']).to('cuda') # [num_candidates, embed_dim]
    
    for sentence in examples:
        print(f"\nInput: {sentence}")
        
        with torch.no_grad():
            features = extract_sentence_features(sentence)
            features_tensor = torch.tensor(features, dtype=cfg['dtype']).unsqueeze(0).to('cuda') # [1, embed_dim]
            h_pos, h_neg = model(features_tensor)
        
        # 计算相似度
        pos_sim = F.cosine_similarity(h_pos, candidate_features, dim=-1) # [batch_size, num_candidates]
        neg_sim = F.cosine_similarity(h_neg, candidate_features, dim=-1) # [batch_size, num_candidates] 
        pos_similarities = pos_sim.cpu().numpy()
        neg_similarities = neg_sim.cpu().numpy()   
             
        # 获取前K个最相似的对象
        top_pos_indices = np.argsort(-pos_similarities)[:top_k]
        print("Top positive objects:")
        for idx in top_pos_indices:
            print(f"  - {candidates[idx]}: {pos_similarities[idx]:.4f}")
        
        top_neg_indices = np.argsort(-neg_similarities)[:top_k]
        print("Top negative objects:")
        for idx in top_neg_indices:
            print(f"  - {candidates[idx]}: {neg_similarities[idx]:.4f}")
        
        print("-"*50)
        
        
def visualize(cfg):
    examples = [
        "This image shows a person, but no motorbike is included.", # p:['person']  n:['motorbike']
        "This image features a car, but no knife is present.", # p:['car']  n:['knife']
        "A bus is included, while a car is absent.", # p:['bus']  n:['car']
        "A bird is present in this image, with no person in sight." # p:['bird']  n:['person']
    ]
    labels = [
        "This image shows a person",
        "This image features a car",
        "A bus is included",
        "A bird is present in this image"
    ]
    candidates = ['bench', 'camera', 'gloves', 'woman', 'screwdriver', 'egg', 'knife', 'plate', 'car', 'motorbike', 'bus', 'bird', 'person']

    """可视化模型预测效果"""
    print("="*50)
    print("CLIP-Lens Prediction Examples")
    print("="*50)
        
    # 提取候选对象文本特征
    candidate_features = []
    for obj in candidates:
        text = f"This image shows a {obj}."
        feature = extract_sentence_features(text)
        candidate_features.append(feature)
    candidate_features = torch.tensor(candidate_features, dtype=cfg['dtype']).to('cuda') # [num_candidates, embed_dim]
    
    for sentence, label in zip(examples, labels):
        print(f"\nInput: {sentence}")
        with torch.no_grad():
            s_features = extract_sentence_features(sentence)
            s_features_tensor = torch.tensor(s_features, dtype=cfg['dtype']).unsqueeze(0).to('cuda')
            l_features = extract_sentence_features(label)
            l_features_tensor = torch.tensor(l_features, dtype=cfg['dtype']).unsqueeze(0).to('cuda')
            # 计算 s_features_tensor 和 l_features_tensor 的相似度
            sim2label = F.cosine_similarity(s_features_tensor, l_features_tensor, dim=-1)
            print(f"  Similarity with label: {sim2label.item():.4f}")
            # 计算 s_features_tensor 和 l_features_tensor 的mse
            mse = F.mse_loss(s_features_tensor, l_features_tensor)
            print(f"  MSE with label: {mse.item():.4f}")
            print("-"*50)
            # s_features_tensor 和 候选对象 的余弦相似度
            s2objs = F.cosine_similarity(s_features_tensor, candidate_features, dim=-1).cpu().numpy() # [batch_size, num_candidates]
            # l_features_tensor 和 候选对象 的余弦相似度
            l2objs = F.cosine_similarity(l_features_tensor, candidate_features, dim=-1).cpu().numpy()
            # 获取前K个最相似的对象
            top_s_indices = np.argsort(-s2objs)[:5]
            top_l_indices = np.argsort(-l2objs)[:5]
            print("Top positive objects:")
            print('----top_s_indices----')
            for idx in top_s_indices:
                print(f"  - {candidates[idx]}: {s2objs[idx]:.4f}") 
            print('----top_l_indices----')   
            for idx in top_l_indices:
                print(f"  - {candidates[idx]}: {l2objs[idx]:.4f}")
            

if __name__ == "__main__":
    # Paths to datasets
    negated_csv = "/root/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    original_csv = "/root/COCO_val_retrieval.csv"
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        
        # -----训练参数-----
        'epochs': 50,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 0.05,
        'early_stop_patience': 5, # Early stopping patience
        'test_only': False, # 是否只测试
        
        # -----数据参数-----
        'pos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
        'negpos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
        'split': [0.9, 0.1, 0.0],  # train, val, test split
        'num_workers': 4,
        'test_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
    }
    
    sys.stdout = Logger(os.path.join(current_dir, "log.txt"))  # 将输出重定向到log.txt文件
    set_seed(3407)  # 设置随机种子为42
    
    # test_dataset = LensDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=cfg['num_workers'])
    
    # if not cfg['test_only']:
    #     model = CLIPGlassesLens(cfg)
    #     # before trainning visualize examples
    #     evaluate(cfg, model, test_loader)
    #     visualize_examples(model, top_k=5)
    #     # Train model
    #     trained_model = train(cfg, model)
    
    # # Final evaluation
    # trained_model = CLIPGlassesLens(cfg)
    # trained_model.load_state_dict(torch.load(os.path.join(current_dir, 'best_clip_lens.pth')))
    # trained_model.eval()
    # trained_model = trained_model.to('cuda')
    # print("\nFinal evaluation on test set:")
    # evaluate(cfg, trained_model, test_loader)
    
    # # after trainning visualize examples
    # visualize_examples(trained_model, top_k=5)
    
    visualize(cfg)