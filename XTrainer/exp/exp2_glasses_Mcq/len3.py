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
import torch.optim as optim

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
        super().__init__()
        embed_dim = Clip_model.visual.output_dim
        hidden_dim = cfg.get('hidden_dim', embed_dim * 2)
        num_heads = cfg.get('num_heads', 4)
        dropout = cfg.get('dropout', 0.1)

        # Self-attention layer (1 layer transformer encoder block)
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        # MLP for gate generation
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )

        # MLP for predicting h_neg
        self.neg_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim)
        )

        # 初始化 gate 为输出接近 0
        self.gate_mlp[-2].bias.data.fill_(-10.0)
        self.neg_mlp[-1].bias.data.zero_()

    def forward(self, h):
        """
        Args:
            h: [batch_size, embed_dim]
        """
        # 模拟 [batch_size, seq_len=1, dim] 进入 transformer encoder
        h_seq = h.unsqueeze(1)  # [B, 1, D]
        h_attended = self.attn_layer(h_seq).squeeze(1)  # [B, D]

        g = self.gate_mlp(h_attended)          # [B, D]
        h_neg = g * self.neg_mlp(h_attended)   # [B, D]
        h_pos = h - h_neg                      # [B, D]

        return h_pos, h_neg


def compute_losses(cfg, h_pos, h_neg, l_pos, h):
    """
    Compute all loss components for CLIPGlassesLens
    
    参数:
        - cfg: 配置参数
        - h_pos: 肯定内容文本特征 [batch_size, embed_dim]
        - h_neg: 否定内容文本特征 [batch_size, embed_dim]
        - l_pos: 监督信号-肯定内容文本特征 [batch_size, embed_dim]
        - h: 原始文本特征 [batch_size, embed_dim]
    
    返回:
        - total_loss: 总损失
        - loss_dict: 各个分项损失的字典 | pos2pos_sim_loss, pos2neg_sim_loss, sparse_loss
    """
    
    # Normalize features for cosine similarity
    h_pos_norm = h_pos / h_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    h_neg_norm = h_neg / h_neg.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    l_pos_norm = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    # h_morm = h / h.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

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
            loss, loss_dict = compute_losses(cfg, h_pos, h_neg, l_pos, h)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['pos2pos_sim_loss'] += loss_dict['pos2pos_sim_loss']
        
        scheduler.step()
        
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
        
        if epoch % 10 == 0:
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
        'epochs': 60,
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
    
    test_dataset = LensDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=cfg['num_workers'])
    
    if not cfg['test_only']:
        model = CLIPGlassesLens(cfg)
        # Train model
        trained_model = train(cfg, model)
    
    # Final evaluation
    trained_model = CLIPGlassesLens(cfg)
    trained_model.load_state_dict(torch.load(os.path.join(current_dir, 'best_clip_lens.pth')))
    trained_model.eval()
    trained_model = trained_model.to('cuda')
    print("\nFinal evaluation on test set:")
    evaluate(cfg, trained_model, test_loader)
    
    # after trainning visualize examples
    visualize_examples(trained_model, top_k=5)
    
    # visualize(cfg)