"""
改动:
    - 给图像侧也使用正交投影，看看效果
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
from sklearn.metrics import average_precision_score
import math

# config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml" # b16
config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB32-ep10-Caltech101-AdamW.yaml"

Clip_model = build_clip_model(config_path=config_path) # 加载CLIP模型

def extract_sentence_features(sentence:str):
    """提取单个句子的CLIP文本特征"""
    with torch.no_grad():  # 关闭梯度计算
        tokenized_text = tokenize(sentence) # [num_classes, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes, context_length]
        text_features = Clip_model.encode_text(tokenized_text) # [num_classes, embed_dim]
        return text_features.cpu().numpy()[0] # [embed_dim]

def extract_img_features(image_path:str):
    """提取单个图像的CLIP图像特征"""
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
            - MCQ任务
    """
    def __init__(self, embed_dim=512, hidden_dim=128, lambda_0=0.8):
        """
        初始化CLIPGlassesFrame模块
        
        Args:
            - embed_dim: 嵌入维度(CLIP特征维度)
            - hidden_dim: MLP隐藏层维度
            - lambda_0: 基础惩罚强度
        """
        super().__init__()
        
        # 温度参数(可学习)
        self.tau = nn.Parameter(torch.ones(1) * 0.07)
        
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
            h_img: 图像特征 [batch_size, embed_dim]
            h: 文本特征 [batch_size, embed_dim]
            
        Returns:
            scores: 匹配得分 [batch_size]
            lambda_dynamic: 动态惩罚权重
        """
        # 图像特征正交投影
        h_img = h_img @ self.W_adpt
        
        # 计算余弦相似度
        cos = F.cosine_similarity(h_img, h_text, dim=1)
        
        # 计算动态惩罚权重
        confidence = self.confidence_mlp(h_text)
        lambda_dynamic = self.lambda_0 * torch.sigmoid(confidence).squeeze(-1)
        
        # 计算标准化差分匹配得分
        scores = (1-lambda_dynamic) * (cos / self.tau) # 67.97%
        
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
        return h, h
    
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
    # state_dict = torch.load(weights_path, map_location=device)
    # model.load_state_dict(state_dict)
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 设置模型为评估模式
    model.eval()
    
    print(f"成功加载 CLIPGlassesLens 模型，权重路径: {weights_path}")
    
    return model


class MCQDataset(Dataset):
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


def train_clip_glasses_frame(cfg):
    """Train the CLIPGlassesFrame model"""
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
    dataset = MCQDataset(cfg.csv_path, Clip_model, lens_model) # Clip_model, lens_model 用于预加载数据过程中的特征提取
    
    # Split dataset into train and validation sets (80/20)
    train_size = int(cfg.train_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # Setup optimizer
    optimizer = optim.AdamW(frame_model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    
    # Training loop
    best_val_acc = 0
    save_dir = Path(cfg.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    for epoch in range(cfg.epochs):
        # Training
        frame_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for img_features, text_features, _, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            img_features = img_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Get scores for all options
            num_options = text_features.shape[1]
            # Process each option
            all_scores = []
            for i in range(num_options):
                scores, _ = frame_model(img_features, text_features[:, i])
                all_scores.append(scores)
            
            # Stack scores for all options
            all_scores = torch.stack(all_scores, dim=1)  # [batch_size, num_options]
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(all_scores, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(all_scores, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        # Validation
        frame_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for img_features, text_features, _,  labels in tqdm.tqdm(val_loader, desc="Validation"):
                img_features = img_features.to(device)
                text_features = text_features.to(device)
                labels = labels.to(device)
                
                # Get scores for all options
                num_options = text_features.shape[1]
                # Process each option
                all_scores = []
                for i in range(num_options):
                    scores, _ = frame_model(img_features, text_features[:, i])
                    all_scores.append(scores)
                
                # Stack scores for all options
                all_scores = torch.stack(all_scores, dim=1)  # [batch_size, num_options]
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(all_scores, labels)
                
                val_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(all_scores, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Print metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(frame_model.state_dict(), save_dir / cfg.frame_weights_path)
            print(f"Saved best model with val acc: {val_acc:.2f}%")
        
        # Save checkpoint
        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': frame_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }
            torch.save(checkpoint, save_dir / f"checkpoint-{epoch+1}-{val_acc}.pth")
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")


def test_clip_glasses_frame(cfg):
    """Test the CLIPGlassesFrame model on a test dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CLIPGlassesLens model
    lens_model = load_clip_glasses_lens(cfg.lens_weights_path, device)
    
    # Initialize and load CLIPGlassesFrame model
    frame_model = CLIPGlassesFrame(embed_dim=512, hidden_dim=128, 
                                lambda_0=cfg.lambda_0).to(device)
    frame_model.load_state_dict(torch.load(os.path.join(cfg.output_dir, cfg.frame_weights_path), map_location=device))
    frame_model.eval()
    
    # Setup test dataset and loader
    test_dataset = MCQDataset(cfg.test_csv_path, Clip_model, lens_model)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    
    # Test loop
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img_features, text_features, _,  labels in tqdm.tqdm(test_loader, desc="Validation"):
            img_features = img_features.to(device)
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            # Get scores for all options
            num_options = text_features.shape[1]
            # Process each option
            all_scores = []
            for i in range(num_options):
                scores, _ = frame_model(img_features, text_features[:, i])
                all_scores.append(scores)
            
            # Stack scores for all options
            all_scores = torch.stack(all_scores, dim=1)  # [batch_size, num_options]
            
            # Compute accuracy
            _, predicted = torch.max(all_scores, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print test metrics
    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),  # 如果 test_acc 是 numpy.float
        'predictions': [int(p) for p in all_predictions],
        'labels': [int(l) for l in all_labels]
    }
    
    with open(cfg.output_dir + "/test_results.json", 'w') as f:
        json.dump(results, f)
    
    print(f"Test results saved to {cfg.output_dir}/test_results.json")
    return test_acc


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
    # 
    
    cfg = {
        # -----模型参数-----
        'lens_weights_path': '/root/NP-CLIP/XTrainer/exp/exp2_glasses_Mcq/len-pretrained/final_clip_lens_b32.pth',  # Path to CLIPGlassesLens weights
        'batch_size': 64,  # Batch size
        'epochs': 5,  # Number of epochs
        'learning_rate': 3e-4,  # Learning rate
        'lambda_0': 0.8,  # Base penalty strength 61.63|b32 62.48%|b16
        
        # -----常规参数-----
        'csv_path': '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv',  # Path to training CSV file
        'train_rate': 0.8,  # Training data ratio
        'num_workers': 4,  # Number of data loading workers
        'output_dir': '/root/NP-CLIP/XTrainer/exp/exp2_glasses_Mcq/Voc2007Mcq-08-frame2-pretrained',  # Output directory for saving models
        'frame_weights_path': f'best_clip_b32_frame2.pth',  # 和 output_dir 拼接得到完整路径
        'test_csv_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv', # Path to test CSV file - 0-shot ACC: 59.73
        'test_only': False,  # Set to True to only run testing
    }
    
    cfg = Config(**cfg)
    
    if not cfg.test_only:
        train_clip_glasses_frame(cfg)
    
    if cfg.test_csv_path:
        if cfg.test_only and not cfg.frame_weights_path:
            raise ValueError("When using --test_only, you must provide --frame_weights_path")
        test_clip_glasses_frame(cfg)


if __name__ == "__main__":
    main()
