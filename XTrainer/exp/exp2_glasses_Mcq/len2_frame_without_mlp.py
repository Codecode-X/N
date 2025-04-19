"""
消融 frame模块的 动态惩罚权重生成器(MLP)：
    - 去除掉了frame模块的 动态惩罚权重生成器(MLP)，改为直接使用可学习的lambda_0参数作为权重
    - 在COCO MCQ数据集上的结果下降到：57.36% -> 39.75%
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
import torch.optim as optim


config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"

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
            - CLIPGlassesLens的输出的否定内容和肯定内容的文本特征h_pos和h_neg。
            - CLIP图像编码器的输出图像特征h_img。
        输出:
            - 文本和图像的匹配度
        场景：
            - MCQ任务
        匹配度计算方法:
            1. **标准化差分匹配得分**  
            对图像特征 $v \in \mathbb{R}^d$ 和文本特征 $h_{pos}, h_{neg}$，定义：  
            $$
            S(v) = \frac{\cos(v, h_{pos})}{\tau_{pos}} - \lambda \cdot \frac{\cos(v, h_{neg})}{\tau_{neg}}
            $$

            - $\tau_{pos}, \tau_{neg}$：温度系数（可学习标量）  
            - $\lambda$：动态惩罚权重（由 $h_{neg}$ 置信度决定，见后）

            2. **动态惩罚权重**  
            $$
            \lambda = \lambda_0 \cdot \sigma\left( \text{MLP}(h_{neg}) \right)
            $$

            - $\lambda_0$：基础惩罚强度（超参数，建议0.5-1.0）  
            - $\sigma(\cdot)$：Sigmoid函数，限制 $\lambda \in (0, \lambda_0)$  
            - MLP：单隐藏层网络（输入维度$d$, 输出维度1）
        
        数据集: COCO_val_mcq_llama3.1_rephrased.csv，格式和示例如下：
            字段：
                - `image_path`：图像路径。
                - `correct_answer`：正确答案的索引（0-3）。
                - `caption_0` ~ `caption_3`：不同的描述，包含肯定、否定或混合表达。
                - `correct_answer_template`：答案类型（`affirmation`、`negation`、`hybrid`）。
            示例：
            image_path	correct_answer	caption_0	caption_1	caption_2	caption_3	correct_answer_template
            data/coco/images/val2017/000000397133.jpg	0	This image features a knife, but no car is present.	A car is present in this image, but there is no knife.	This image features a car	This image does not feature a knife.	hybrid
            data/coco/images/val2017/000000397133.jpg	0	A chair is not present in this image.	This image shows a chair, but no spoon is present.	A chair is present in this image.	A spoon is not included in this image.	negative
            data/coco/images/val2017/000000397133.jpg	0	A person is present in this image, but there's no fork.	This image shows a fork, with no person in sight.	A fork is shown in this image.	No person is present in this image.	hybrid

        训练策略：
        
            #### 一、数据预处理流程

            1. **特征预提取**（离线完成，仅需一次）：
            - 对每个候选描述 $c_i (i=0,1,2,3)$：
                - 使用 CLIP-GlassesLens 提取 $h_{pos}^{(i)}, h_{neg}^{(i)}$
            - 对图像 $v$：
                - 使用原始 CLIP 提取 $h_{img}$

            2. **标签构建**：
            - 对每个样本生成 One-Hot 标签向量 $y \in \{0,1\}^4$（正确选项为1）

            ---

            #### 二、模型训练配置

            | 组件           | 配置                                             | 说明                                  |
            | -------------- | ------------------------------------------------ | ------------------------------------- |
            | **可训练参数** | MLP参数、$\tau_{pos}$, $\tau_{neg}$, $\lambda_0$ | 总参数量 <1K                          |
            | **优化器**     | AdamW                                            | $lr=3\times10^{-4}, \beta=(0.9,0.98)$ |
            | **Batch Size** | 64                                               | 显存不足时可降至32                    |
            | **训练轮数**   | 50                                               | 监控早停                              |

            ---

            #### 三、损失函数设计

            1. **匹配度计算**：
            $$
            S_i = \frac{\cos(h_{img}, h_{pos}^{(i)})}{\tau_{pos}} - \lambda^{(i)} \cdot \frac{\cos(h_{img}, h_{neg}^{(i)})}{\tau_{neg}}
            $$

            - 每个候选描述计算得分 $S_0, S_1, S_2, S_3$

            2. **交叉熵损失**：
            $$
            \mathcal{L}_{\text{CE}} = -\sum_{i=0}^3 y_i \log\left(\frac{e^{S_i}}{\sum_j e^{S_j}}\right)
            $$

            3. **正则化项**：
            $$
            \mathcal{L}_{\text{reg}} = 0.05 \cdot \|\text{MLP}(h_{neg})\|_2^2
            $$
        """
    def __init__(self, embed_dim=512, hidden_dim=128, lambda_0=0.8):
        """
        初始化CLIPGlassesFrame模块
        
        Args:
            embed_dim: 嵌入维度(CLIP特征维度)
            hidden_dim: MLP隐藏层维度
            lambda_0: 基础惩罚强度
        """
        super().__init__()
        
        # 温度参数(可学习)
        self.tau_pos = nn.Parameter(torch.ones(1) * 0.07)
        self.tau_neg = nn.Parameter(torch.ones(1) * 0.07)
        self.lambda_neg = nn.Parameter(torch.ones(1) * lambda_0)
        
        # 用于动态惩罚权重的MLP
        self.confidence_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.lambda_0 = lambda_0
        self.reg_weight = 0.05  # 正则化损失权重
        
    def forward(self, h_img, h_pos, h_neg):
        """
        计算图像和文本特征的匹配得分
        
        Args:
            h_img: 图像特征 [batch_size, embed_dim]
            h_pos: 肯定文本特征 [batch_size, embed_dim]
            h_neg: 否定文本特征 [batch_size, embed_dim]
            
        Returns:
            scores: 匹配得分 [batch_size]
        """
        # 计算余弦相似度
        cos_pos = F.cosine_similarity(h_img, h_pos, dim=1)
        cos_neg = F.cosine_similarity(h_img, h_neg, dim=1)
        
        # 计算标准化差分匹配得分
        scores = cos_pos / self.tau_pos - self.lambda_neg * (cos_neg / self.tau_neg)
        
        return scores
    
    def compute_mcq_scores(self, h_img, h_pos_list, h_neg_list):
        """
        计算多选题(MCQ)任务的得分
        
        Args:
            h_img: 图像特征 [batch_size, embed_dim]
            h_pos_list: 多个选项的肯定特征 [num_options, batch_size, embed_dim]
            h_neg_list: 多个选项的否定特征 [num_options, batch_size, embed_dim]
            
        Returns:
            all_scores: 所有选项的得分 [batch_size, num_options]
        """
        num_options = len(h_pos_list)
        batch_size = h_img.shape[0]
        device = h_img.device
        
        all_scores = torch.zeros(batch_size, num_options, device=device)
        
        for i in range(num_options):
            scores = self.forward(h_img, h_pos_list[i], h_neg_list[i])
            all_scores[:, i] = scores
            
        return all_scores
        

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
    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到模型权重文件: {weights_path}")
    
    # 默认配置
    config = {
        'embed_dim': 512,  # CLIP 文本特征的维度
        'hidden_dim': 256   # 隐藏层维度
    }
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = CLIPGlassesLens(config)
    
    # 加载权重
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    
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
        self.data = pd.read_csv(csv_path)
        self.clip_model = clip_model
        self.lens_model = lens_model
        self.transform = transform
        self.device = next(lens_model.parameters()).device
        
        # Preprocess all data
        self._preprocess_features()
        
    def _preprocess_features(self):
        """Preprocess and cache all image and text features"""
        print("Preprocessing dataset features...")
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
    frame_model = CLIPGlassesFrame(embed_dim=512, hidden_dim=128, lambda_0=cfg.lambda_0).to(device)

    # Setup dataset and dataloader
    dataset = MCQDataset(cfg.csv_path, Clip_model, lens_model)
    
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
        
        for img_features, pos_features, neg_features, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            img_features = img_features.to(device)
            pos_features = pos_features.to(device)
            neg_features = neg_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Get scores for all options
            batch_size = img_features.shape[0]
            num_options = pos_features.shape[1]
            
            # Process each option
            all_scores = []
            
            for i in range(num_options):
                option_pos = pos_features[:, i]
                option_neg = neg_features[:, i]
                
                scores = frame_model(img_features, option_pos, option_neg)
                all_scores.append(scores)
            
            # Stack scores for all options
            all_scores = torch.stack(all_scores, dim=1)  # [batch_size, num_options]
            
            # Compute cross-entropy loss
            ce_loss = F.cross_entropy(all_scores, labels)
            loss = ce_loss
            
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
            for img_features, pos_features, neg_features, labels in tqdm.tqdm(val_loader, desc="Validation"):
                img_features = img_features.to(device)
                pos_features = pos_features.to(device)
                neg_features = neg_features.to(device)
                labels = labels.to(device)
                
                # Process each option
                all_scores = []
                
                for i in range(pos_features.shape[1]):
                    option_pos = pos_features[:, i]
                    option_neg = neg_features[:, i]
                    
                    scores = frame_model(img_features, option_pos, option_neg)
                    all_scores.append(scores)
                
                # Stack scores for all options
                all_scores = torch.stack(all_scores, dim=1)  # [batch_size, num_options]
                
                # Compute cross-entropy loss
                ce_loss = F.cross_entropy(all_scores, labels)
                loss = ce_loss
                
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
        
        # 打印frame学到的lambda_0, tau_pos, tau_neg
        print(f"lambda_0: {frame_model.lambda_neg.item():.4f}, tau_pos: {frame_model.tau_pos.item():.4f}, tau_neg: {frame_model.tau_neg.item():.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(frame_model.state_dict(), save_dir / "best_clip_frame.pth")
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
    frame_model = CLIPGlassesFrame(embed_dim=512, hidden_dim=128, lambda_0=cfg.lambda_0).to(device)
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
        for img_features, pos_features, neg_features, labels in tqdm.tqdm(test_loader, desc="Testing"):
            img_features = img_features.to(device)
            pos_features = pos_features.to(device)
            neg_features = neg_features.to(device)
            labels = labels.to(device)
            
            # Process each option
            all_scores = []
            
            for i in range(pos_features.shape[1]):
                option_pos = pos_features[:, i]
                option_neg = neg_features[:, i]
                
                scores = frame_model(img_features, option_pos, option_neg)
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
    
    cfg = {
        # 'csv_path': '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv',  # Path to training CSV file
        'csv_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',  # Path to training CSV file
        'train_rate': 0.8,  # Training data ratio
        'lens_weights_path': '/root/NP-CLIP/XTrainer/exp/exp2_glasses_Mcq/len-pretrained/final_clip_lens.pth',  # Path to CLIPGlassesLens weights
        'batch_size': 64,  # Batch size
        'epochs': 100,  # Number of epochs
        'learning_rate': 3e-4,  # Learning rate
        'lambda_0': 2,  # Base penalty strength
        'num_workers': 4,  # Number of data loading workers
        'output_dir': '/root/NP-CLIP/XTrainer/exp/exp2_glasses_Mcq/glasses-CocoMcq-08-frame-pretrained',  # Output directory for saving models
        'frame_weights_path': 'best_clip_frame.pth',  # 和 output_dir 拼接得到完整路径
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
