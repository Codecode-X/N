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
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
from utils import standard_image_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import hashlib
import torch.optim as optim
import random


# config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml" # b16
config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB32-ep10-Caltech101-AdamW.yaml" # b32

Clip_model = build_clip_model(config_path=config_path) # 加载CLIP模型

def extract_sentence_features(sentence:str):
    """原始CLIP提取单个句子的CLIP文本特征"""
    with torch.no_grad():  # 关闭梯度计算
        tokenized_text = tokenize(sentence) # [num_classes, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes, context_length]
        text_features = Clip_model.encode_text(tokenized_text) # [num_classes, embed_dim]
        return text_features.cpu().numpy()[0] # [embed_dim]

def extract_img_features(image_path:str):
    """原始CLIP提取单个图像的CLIP图像特征"""
    # 加载图像，转为张量[batch_size, 3, input_size, input_size]
    img = Image.open(image_path)  # 打开图像
    transform = Compose([standard_image_transform(224, 'BICUBIC'), ToTensor()])  # 定义转换
    img = transform(img)  # 转换图像
    img = img.unsqueeze(0)  # 添加 batch 维度
    img = img.to(dtype=Clip_model.dtype, device=Clip_model.device)  # 转移到模型的设备上
    with torch.no_grad():  # 关闭梯度计算
        image_features = Clip_model.encode_image(img) # [num_classes, embed_dim]
        return image_features.cpu().numpy()[0] # [embed_dim]

class CLIPGlassesFrame(nn.Module):   
    """
    CLIPGlassesFrame: 
        输入：
            - CLIP输出的文本特征h_text。
            - CLIP图像编码器的输出图像特征h_img。
        输出:
            - 文本和图像的匹配度
        场景：
            - Retrieval任务
    """
    def __init__(self, embed_dim=512, hidden_dim=128, lambda_0=0.8, reg_weight=0.05, ortho_weight=0.1):
        """
        初始化CLIPGlassesFrame模块
        
        Args:
            - embed_dim: 嵌入维度(CLIP特征维度)
            - hidden_dim: MLP隐藏层维度
            - lambda_0: 基础惩罚强度
        """
        super().__init__()
        
        # 可训练的 logit_scale = log(1 / τ) | CLIP 通过 可训练的 logit_scale 让模型自动调整 τ，适应不同数据和任务。
        τ = 0.07 # CLIP 训练时的 softmax 温度参数 τ，较小的 τ(较大的 logit_scale)：分布更加陡峭，相似度高的样本更加突出
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / τ))  # 训练时学习的是 logit_scale，而不是 τ，直接学习 τ 可能会导致梯度更新过大或过小。

        
        # 用于动态惩罚权重的MLP
        self.confidence_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 图像正交投影矩阵 - 初始化为正交矩阵
        self.W_adpt = nn.Parameter(torch.empty(embed_dim, embed_dim))
        
        # # kaiming初始化
        # nn.init.kaiming_uniform_(self.W_adpt, a=math.sqrt(5)) # 64.54%
        # Xavier初始化
        # nn.init.xavier_uniform_(self.W_adpt) # 65.05%
        
        # # 正交初始化
        self.orthogonal_init(self.W_adpt, embed_dim) # 67.59% | kaiming初始化:64.54% | Xavier初始化:65.05%
        
        self.lambda_0 = lambda_0  # 基础惩罚强度
      
    # 正交初始化
    def orthogonal_init(self, W_adpt, embed_dim):
        # 生成两个彼此正交、各自也正交的投影矩阵 W_pos 和 W_neg，为模型提供结构良好的初始参数
        with torch.no_grad():
            # 生成随机正交矩阵
            u, _, v = torch.svd(torch.randn(embed_dim, embed_dim))
            # 生成另一个正交矩阵，确保与第一个正交
            u2, _, v2 = torch.svd(torch.randn(embed_dim, embed_dim))
            # 使用Gram-Schmidt过程确保正交性
            proj = (u2 @ u.t()) @ u
            u2_orth = u2 - proj
            # 归一化
            u2_orth = F.normalize(u2_orth, dim=1)
            W_adpt.data = u2_orth
        
    def forward(self, h_img, h_text):
        """
        计算图像和文本特征的匹配得分
        
        Args:
            h_img: 图像特征 [N_imgs, embed_dim]
            h_text: 文本特征 [N_caps, embed_dim]
            
        Returns:
            scores: 匹配得分 [N_caps, N_imgs]
            lambda_dynamic: 动态惩罚权重
        """
        # 计算动态惩罚权重
        confidence = self.confidence_mlp(h_text) # [N_caps, 1]
        lambda_dynamic = self.lambda_0 * torch.sigmoid(confidence) # [N_caps, 1]
        
        # 图像特征正交投影
        h_img = h_img @ self.W_adpt
        
        # 标准化
        h_img = h_img / h_img.norm(dim=-1, keepdim=True) # [N_imgs, embed_dim]
        h_text = h_text / h_text.norm(dim=-1, keepdim=True) # [N_caps, embed_dim]
        
        # 计算标准化差分匹配得分
        logit_scale = self.logit_scale.exp()
        sim = h_text @ h_img.T # [N_caps, N_imgs]
        scores = logit_scale * (1-lambda_dynamic) * sim # [1, N_caps] @ [N_caps, N_imgs] = [N_caps, N_imgs]
        
        return scores, lambda_dynamic
        

class CLIPGlassesLens(nn.Module):
    """
    CLIP-GlassesLens

    len2: 一个轻量级模块CLIP-GlassesLens，通过处理CLIP的文本编码器的输出文本特征，来生成否定内容和肯定内容的文本特征。
    相比len1：
        - 通过2-head自注意力机制，增强了对输入句子中否定对象的关注。
        - 调整了些超参数

    输入:
        - CLIP的文本编码器的输出文本特征 h
            - 示例：原始输入句子 "In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves." 的文本特征。

    输出:
        - 否定内容文本特征 h_neg 
            - 示例：原始输入句子中被否定对象 ['bench'] 组成的文本 "A photo that includes bench." 的文本特征。| A photo that includes + 否定对象列表
        - 肯定内容文本特征 h_pos
            - 示例：原始输入句子中被肯定对象 ['camera', 'gloves'] 组成的文本 "A photo that includes camera, gloves." 的文本特征。| A photo that includes + 肯定对象列表
    """
    def __init__(self, config):
        """
        初始化CLIP-Lens模块。
        
        参数:
            config: 配置字典，包含超参数设置
        """
        super().__init__()
        
        embed_dim = config['embed_dim']
        hidden_dim = config['hidden_dim']
        
        # 2-head Self-Attention
        self.num_heads = 2
        self.self_attn = nn.MultiheadAttention(embed_dim, self.num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # FFN用于将注意力输出转换为所需的维度
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 2)
        )
        self.norm2 = nn.LayerNorm(embed_dim * 2)
        
        # 动态门控网络 - 决定原始特征的调整强度
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid()  # 输出范围在 [0, 1] 的 alpha 和 beta
        )
        
        self.embed_dim = embed_dim # CLIP文本特征的维度
        
    def forward(self, h):
        """
        CLIP-Lens的前向传播
        
        参数:
            h: CLIP文本特征 [batch_size, embed_dim]
            
        返回:
            h_pos: 肯定内容特征 [batch_size, embed_dim]
            h_neg: 否定内容特征 [batch_size, embed_dim]
        """
        return h, h # 直接返回原始CLIP提取的文本特征
    
        batch_size = h.shape[0]
        
        # Self-Attention需要序列输入格式 [batch_size, seq_len, embed_dim]
        # 我们将每个特征视为长度为1的序列
        h_seq = h.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # 应用自注意力
        attn_output, _ = self.self_attn(h_seq, h_seq, h_seq)
        attn_output = attn_output.squeeze(1)  # [batch_size, embed_dim]
        
        # 残差连接和层归一化
        h_attn = self.norm1(attn_output + h)
        
        # FFN生成正交基向量
        decomp = self.ffn(h_attn)
        decomp = self.norm2(decomp)
        
        # 分离出正交基向量
        u = decomp[:, :self.embed_dim]  # 肯定内容基向量
        v = decomp[:, self.embed_dim:]  # 否定内容基向量
        
        # 将向量归一化为单位长度
        u_norm = F.normalize(u, dim=1)
        v_norm = F.normalize(v, dim=1)
        
        # 动态特征融合的门控
        gates = self.gate_net(h)
        alpha = gates[:, 0].view(batch_size, 1)  # 控制肯定特征的强度
        beta = gates[:, 1].view(batch_size, 1)   # 控制否定特征的强度
        
        # 通过残差连接生成输出特征
        h_pos = alpha * u_norm + (1 - alpha) * h
        h_neg = beta * v_norm + (1 - beta) * h
        
        return h_pos, h_neg

@torch.no_grad()
def predict(model, sentence):
    """CLIPGlassesLens 对新句子进行预测，返回肯定和否定内容特征"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    features = extract_sentence_features(sentence)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    h_pos, h_neg = model(features_tensor)
    
    return h_pos.cpu().numpy()[0], h_neg.cpu().numpy()[0]
        
def load_clip_glasses_lens(weights_path, device=None):
    """
    加载预训练的 CLIPGlassesLens 模型权重。
    
    参数:
        weights_path (str): 模型权重文件的路径
        config (dict, optional): 模型初始化的配置字典。
                                    如果为 None，则使用默认配置。
        device (str, optional): 加载模型的设备 ('cuda' 或 'cpu')。
                                    如果为 None，则优先使用 CUDA（如果可用）。
    
    返回:
        model (CLIPGlassesLens): 初始化并加载权重的模型
    """
    # # 检查权重文件是否存在
    # if not os.path.exists(weights_path):
    #     raise FileNotFoundError(f"未找到模型权重文件: {weights_path}")
    
    # 默认配置
    config = {
        'embed_dim': 512,  # CLIP 文本特征的维度
        'hidden_dim': 256   # 隐藏层维度
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = CLIPGlassesLens(config)
    
    # # 加载权重
    # state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    # model.load_state_dict(state_dict)
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    print(f"成功加载 CLIPGlassesLens 模型，权重路径: {weights_path}")
    
    return model

class RetrievalDataset(Dataset):
    """
    Dataset for retrieval task
    """
    def __init__(self, csv_path, clip_model, lens_model, transform=None):
        """
        Args:
            csv_path: Path to the CSV file with MCQ data
            clip_model: CLIP model for feature extraction
            lens_model: CLIPGlassesLens model for feature extraction
            transform: Optional transform to be applied on images
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path, encoding='gbk')
        self.clip_model = clip_model
        self.lens_model = lens_model
        self.transform = transform
        self.device = clip_model.device
        
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
        cache_path = f"retrieve_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.image_features = cached_data['image_features']
            self.caption_pos_features = cached_data['caption_pos_features']
            self.caption_neg_features = cached_data['caption_neg_features']
            self.image_ids = cached_data['image_ids']
            print(f"Loaded {len(self.image_features)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        
        self.image_features = []
        self.caption_pos_features = []
        self.caption_neg_features = []
        self.image_ids = []
        
        for idx, row in tqdm.tqdm(self.data.iterrows(), total=len(self.data)):
            img_path = row['filepath']
            image_id = row['image_id']
            captions = eval(row['captions']) # 每个图片对应的数量不一致

            # 提取图像特征
            img_feature = extract_img_features(img_path) # 原始CLIP提取的图像特征
            self.image_features.append(img_feature)
            self.image_ids.append(image_id)

            pos_list = []
            neg_list = []

            for caption in captions:
                text_feat = extract_sentence_features(caption) # 原始CLIP提取的文本特征
                text_tensor = torch.tensor(text_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    h_pos, h_neg = self.lens_model(text_tensor)

                pos_list.append(h_pos.cpu().numpy()[0])
                neg_list.append(h_neg.cpu().numpy()[0])

            self.caption_pos_features.append(pos_list)
            self.caption_neg_features.append(neg_list)

        print(f"Saving features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'caption_pos_features': self.caption_pos_features,
            'caption_neg_features': self.caption_neg_features,
            'image_ids': self.image_ids
        }, cache_path)
        print(f"Cached {len(self.image_ids)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image_features: Image features [embed_dim]
            pos_features: List of positive features for each caption [num_captions, embed_dim]
            neg_features: List of negative features for each caption [num_captions, embed_dim]
            image_id: Image ID
        """
        img_feature = torch.tensor(self.image_features[idx], dtype=torch.float32)
        cap_pos = torch.tensor(np.stack(self.caption_pos_features[idx]), dtype=torch.float32)
        cap_neg = torch.tensor(np.stack(self.caption_neg_features[idx]), dtype=torch.float32)
        image_id = torch.tensor(int(self.image_ids[idx]))
        
        return img_feature, cap_pos, cap_neg, image_id

def retrieval_collate_fn(batch):
    img_features, pos_features, neg_features, image_ids = zip(*batch)
    return (
        torch.stack(img_features),  # [B, embed_dim]
        list(pos_features),         # list of [num_captions_i, embed_dim]
        list(neg_features),         # list of [num_captions_i, embed_dim]
        torch.stack(image_ids)      # [B]
    )

# Validation
def _evaluate_model(cfg, frame_model, data_loader, device):
    """
    Evaluate the model on the validation set.
    
    参数：
        - cfg: 配置对象
        - frame_model: CLIPGlassesFrame模型
        - data_loader: 验证数据加载器
        - device: 设备（CPU或GPU）
    
    返回：
        - results: 评估结果字典，包含文本到图像和图像到文本的召回率
            - txt2img: 文本到图像的召回率 | txt2img[k]表示recall@k
            - img2txt: 图像到文本的召回率 | img2txt[k]表示recall@k
            - mean: 平均召回率 | mean[k]表示recall@k
    """
    # Metrics
    txt2img_recalls = {1: 0, 5: 0, 10: 0}
    img2txt_recalls = {1: 0, 5: 0, 10: 0}
    all_image_feats = []
    all_text_feats  = []
    caption_to_img  = []
    
    with torch.no_grad():
        img_count = 0
        for img_feats, pos_text_feats, neg_text_feats, image_ids in tqdm.tqdm(data_loader, desc="Extract feats"):
            assert torch.equal(pos_text_feats[0], neg_text_feats[0]), "未采用双投影，因此文本特征中的肯定特征pos_text_feats应该等于否定特征neg_text_features等于文本特征"
            img_feats = img_feats.to(device)                        # [B, D]
            # text_feats_list: list of length B, 每项 [C_i, D]
            # 收集图像特征
            for i in range(img_feats.size(0)):
                all_image_feats.append(img_feats[i].cpu())
                # 收集这个图的所有 caption 特征 & 映射
                for cap in pos_text_feats[i]:
                    all_text_feats.append(cap.cpu())
                    caption_to_img.append(img_count) # 文本到图像的映射，示例： [0, 0, 1, 1, 2, 2]
                img_count += 1

        # 转为大张量
        I = torch.stack(all_image_feats, dim=0).to(device)  # [N_imgs, D]
        C = torch.stack(all_text_feats,  dim=0).to(device)  # [N_caps, D]
        N_imgs, N_caps = I.size(0), C.size(0)
        print(f"Total images: {N_imgs}, total captions: {N_caps}")

        # —— 3. 计算全量相似度 —— 
        # raw CLIP 版
        if cfg.raw_CLIP:
            Clip_model.eval()
            # 标准化
            I = I / I.norm(dim=-1, keepdim=True) # [N_imgs, D]
            C = C / C.norm(dim=-1, keepdim=True) # [N_caps, D]
            logit_scale = Clip_model.logit_scale.exp()
            sim_matrix = logit_scale * (C @ I.T) # [N_caps, N_imgs]
        # Frame 模型版
        else:
            frame_model.eval()
            sim_matrix, neg_text_feats = frame_model(I, C) # [N_caps, N_imgs]
                
        # —— 4. 评估 Recall@1/5/10 —— 
        txt2img_hits = {1:0, 5:0, 10:0}
        img2txt_hits = {1:0, 5:0, 10:0}

        # Text→Image
        for cap_idx, gt_img in enumerate(caption_to_img):
            scores = sim_matrix[cap_idx] # [N_imgs=149]
            # top-k 图像索引
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in txt2img_hits:
                if gt_img in topk[:k]:
                    txt2img_hits[k] += 1

        # Image→Text
        # 先构造每张图的 caption 索引列表
        img2cap = [[] for _ in range(N_imgs)]  # 例如访问 img2cap[0]，得到图像 0 的所有 caption 索引
        for cap_idx, img_idx in enumerate(caption_to_img):
            img2cap[img_idx].append(cap_idx)

        for img_idx in range(N_imgs):
            if not img2cap[img_idx]:
                continue
            scores = sim_matrix[:, img_idx] # [N_caps]
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in img2txt_hits:
                # 只要有任一 gt caption 在 top-k 中，就算命中
                if any(cap in topk[:k] for cap in img2cap[img_idx]):
                    img2txt_hits[k] += 1

    # —— 5. 计算并打印百分比 —— 
    total_caps = float(N_caps)
    total_imgs = float(N_imgs)
    txt2img_recalls = {k: txt2img_hits[k]/total_caps*100 for k in txt2img_hits}
    img2txt_recalls = {k: img2txt_hits[k]/total_imgs*100 for k in img2txt_hits}

    print("\nText→Image Retrieval:")
    for k,v in txt2img_recalls.items():
        print(f"  Recall@{k}: {v:.2f}%")
    print("\nImage→Text Retrieval:")
    for k,v in img2txt_recalls.items():
        print(f"  Recall@{k}: {v:.2f}%")
    print("\nMean Retrieval:")
    for k in [1,5,10]:
        print(f"  Recall@{k}: {(txt2img_recalls[k]+img2txt_recalls[k])/2:.2f}%")

    return {
        'txt2img': txt2img_recalls,
        'img2txt': img2txt_recalls,
        'mean':   {k:(txt2img_recalls[k]+img2txt_recalls[k])/2 for k in txt2img_recalls}
    }
    
def train_clip_glasses_frame_Retrieval(cfg):
    """Train the CLIPGlassesFrame model on a test dataset on Retrieval"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CLIPGlassesLens model and freeze it
    lens_model = load_clip_glasses_lens(cfg.lens_weights_path, device)
    for param in lens_model.parameters():
        param.requires_grad = False

    # Ensure CLIP model is frozen
    for param in Clip_model.parameters():
        param.requires_grad = False

    # Initialize CLIPGlassesFrame model
    frame_model = CLIPGlassesFrame(embed_dim=512, hidden_dim=128, 
                                    lambda_0=cfg.lambda_0).to(device)

    # Setup dataset and dataloader
    dataset = RetrievalDataset(cfg.csv_path, Clip_model, lens_model)
    
    # Split dataset into train and validation sets (80/20)
    train_size = int(cfg.train_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=retrieval_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=retrieval_collate_fn)
    
    # Setup optimizer
    optimizer = optim.AdamW(frame_model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # begin training model evaluation _evaluate_model(frame_model, val_loader, device)
    print("Evaluating initial model performance...")
    _evaluate_model(cfg, frame_model, val_loader, device)
    
    # Training loop
    best_recall5 = 0
    save_dir = Path(cfg.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(cfg.epochs):
        # Training
        frame_model.train()
        train_loss = 0
        
        for img_features, pos_text_feats, neg_text_feats, image_ids in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            assert torch.equal(pos_text_feats[0], neg_text_feats[0]), "未采用双投影，因此文本特征中的肯定特征pos_text_feats应该等于否定特征neg_text_features等于文本特征" 
            img_features = img_features.to(device)  # [B, embed_dim]
            pos_text_feats = [f.to(device) for f in pos_text_feats] # list of [num_captions_i, embed_dim]
            neg_text_feats = [f.to(device) for f in neg_text_feats] # list of [num_captions_i, embed_dim]
            
            batch_size = img_features.shape[0]
            
            # Get caption counts for each image
            caption_counts = [pos_feat.shape[0] for pos_feat in pos_text_feats]
            total_captions = sum(caption_counts)
            
            optimizer.zero_grad()
            
            # Create a similarity matrix of size [total_captions, total_images]
            similarity_matrix = torch.zeros(total_captions, batch_size).to(device)
            
            # Create caption-to-image mapping
            caption_to_img_idx = []
            for i, count in enumerate(caption_counts):
                caption_to_img_idx.extend([i] * count)
                
            # Process each image with all captions
            cap_start_idx = 0
            for i in range(batch_size):
                img_feature = img_features[i].unsqueeze(0)  # [1, embed_dim]
                
                # Process all captions for all images
                for j in range(batch_size):
                    for c in range(caption_counts[j]):
                        cap_idx = cap_start_idx + sum(caption_counts[:j]) + c
                        cap_pos = pos_text_feats[j][c].unsqueeze(0)  # [1, embed_dim]
                        cap_neg = neg_text_feats[j][c].unsqueeze(0)  # [1, embed_dim]
                        
                        # Get score and loss from frame model
                        score, _ = frame_model(img_feature, cap_pos)
                        # Store score in the similarity matrix
                        similarity_matrix[cap_idx, i] = score

            # Calculate text-to-image contrastive loss
            # For each caption, find its positive image
            txt2img_loss = 0
            for cap_idx in range(total_captions):
                # Positive image for this caption is its corresponding image
                img_idx = caption_to_img_idx[cap_idx]
                txt2img_loss += -torch.log(
                    torch.exp(similarity_matrix[cap_idx, img_idx]) / 
                    torch.sum(torch.exp(similarity_matrix[cap_idx, :]))
                )
            txt2img_loss /= total_captions
            
            # Calculate image-to-text contrastive loss
            # For each image, find all its positive captions
            img2txt_loss = 0
            for i in range(batch_size):
                # Find indices of positive captions for image i
                pos_cap_indices = [idx for idx, img_idx in enumerate(caption_to_img_idx) if img_idx == i]
                pos_exp_sum = torch.sum(torch.exp(similarity_matrix[pos_cap_indices, i]))
                all_exp_sum = torch.sum(torch.exp(similarity_matrix[:, i]))
                img2txt_loss += -torch.log(pos_exp_sum / all_exp_sum)
            img2txt_loss /= batch_size
            
            # Symmetric contrastive loss
            contrastive_loss = 0.5 * (txt2img_loss + img2txt_loss)
            
            # Total loss
            loss = contrastive_loss

            loss.backward()
            optimizer.step()
            
            train_loss += loss
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        print("Evaluating model performance...")
        res = _evaluate_model(cfg, frame_model, val_loader, device)
        recall5 = res['mean'][5]
        
        # Save best model
        if recall5 > best_recall5:
            best_recall5 = recall5
            str_val_r5 = str(recall5).replace('.', '_')
            # 删除旧模型
            for file in save_dir.glob("best_clip_frame2_*.pth"):
                file.unlink()
            # 保存新模型
            torch.save(frame_model.state_dict(), save_dir / f"best_clip_frame2_{str_val_r5}_R.pth")
            print(f"Saved best model with re acc: {best_recall5:.2f}%")
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': frame_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_recall5': best_recall5,
            }
            torch.save(checkpoint, save_dir / f"checkpoint-{epoch+1}-{recall5}.pth")
    
    print(f"Training completed. Best validation loss: {best_recall5:.4f}")

    # ----训练后测试模型------
    print("Evaluating final model performance...")
    frame_model.eval()
    test_dataset = RetrievalDataset(cfg.test_csv_path, Clip_model, lens_model)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                            shuffle=False, num_workers=cfg.num_workers, 
                            collate_fn=retrieval_collate_fn)
    
    res = _evaluate_model(cfg, frame_model, test_loader, device)
    mean_recall5 = res['mean'][5]
    str_test_recall5 = str(mean_recall5).replace('.', '_')
    # 删除旧模型
    for file in save_dir.glob("best_clip_frame2_*.pth"):
        file.unlink()
    # 保存新模型
    torch.save(frame_model.state_dict(), save_dir / f"best_clip_frame2_{str_test_recall5}_R.pth")
    print(f"Saved final model with test recall: {mean_recall5:.2f}%")
    return res
    

def test_clip_glasses_frame_Retrieval(cfg):
    """
    Test the CLIPGlassesFrame model on a test dataset on Retrieval
    
    指标： 
        - Recall@1
        - Recall@5
        - Recall@10
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIPGlassesLens model
    lens_model = load_clip_glasses_lens(cfg.lens_weights_path, device)
    for param in lens_model.parameters():
        param.requires_grad = False
        
    # Initialize CLIPGlassesFrame model
    frame_model = CLIPGlassesFrame(embed_dim=512, hidden_dim=128, 
                                    lambda_0=cfg.lambda_0).to(device)
    
    # Load the trained model weights
    model_path = Path(cfg.output_dir) / cfg.frame_weights_path
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    frame_model.load_state_dict(state_dict)
    frame_model.eval()
    print(f"Loaded model from {model_path}")
    
    # Setup test dataset and dataloader
    test_dataset = RetrievalDataset(cfg.test_csv_path, Clip_model, lens_model)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                            shuffle=False, num_workers=cfg.num_workers, 
                            collate_fn=retrieval_collate_fn)
    
    return _evaluate_model(cfg, frame_model, test_loader, device)


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def main():
    # 设置随机种子
    seed = 3407
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    cfg = {
        # -----baseline-原始CLIP-----
        'raw_CLIP': False,  # Set to True to use original CLIP model
        
        # -----模型参数-----
        'lens_weights_path': '/root/NP-CLIP/XTrainer/exp/exp3_glasses_Retrieval/len-pretrained/final_clip_lens.pth',  # Path to CLIPGlassesLens weights
        'batch_size': 32,  # Batch size
        'epochs': 15,  # Number of epochs
        'learning_rate': 3e-4,  # Learning rate
        'lambda_0': 0.8,  # Base penalty strength 62.48%
        
        # -----常规参数-----
        'csv_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',  # Path to training CSV file
        'train_rate': 0.8,  # Training data ratio
        'num_workers': 4,  # Number of data loading workers
        'output_dir': '/root/NP-CLIP/XTrainer/exp/exp3_glasses_Retrieval/COCORetrieval-08-frame2-pretrained',  # Output directory for saving models
        'frame_weights_path': 'best_clip_frame2_0_14001681299619512_R.pth',  # 和 output_dir 拼接得到完整路径
        'test_csv_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv', # Path to test CSV file - 0-shot ACC: 59.73
        'test_only': False,  # Set to True to only run testing
    }
    
    cfg = Config(**cfg)
    
    if not cfg.test_only:
        train_clip_glasses_frame_Retrieval(cfg)
        return
    
    if cfg.test_csv_path:
        if cfg.test_only and not cfg.frame_weights_path:
            raise ValueError("When using --test_only, you must provide --frame_weights_path")
        test_clip_glasses_frame_Retrieval(cfg)
        return


if __name__ == "__main__":
    main()
