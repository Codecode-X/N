"""
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
    
训练： 
    - 训练数据集：MCQ_natural_polished.csv
    - 监督信号: 
        - 1. 训练数据集中每个输入句子中包含的否定对象组 L_n 组成的文本，并经过CLIP文本编码器的输出文本特征 l_neg
        - 2. 训练数据集中每个输入句子中包含的肯定对象组 L_p 组成的文本，并经过CLIP文本编码器的输出文本特征 l_pos
    - 通过对比学习，训练轻量级模块，使得 h_neg 和 l_neg 的相似度更高，而 h_pos 和 l_pos 的相似度更高，同时 h_neg 和 h_pos 的相似度更低。

数据集： /root/NP-CLIP/NegBench/data/MCQ_natural_polished.csv

    i	n	m	S_pn	L_p	L_n
    1	2	1	In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves.	['bench']	['camera', 'gloves']
    2	0	3	In a cozy bakery, a fluffy lion, an elegant coffee table, and an egg share the space, though 	['lion', 'coffee table', 'egg']	[]
    ...
    1000	2	4	On a wooden dining table amidst a quiet afternoon, you can see a bright gloves, a woman, a screwdriver, a delicious egg, and yet without a knife and no a plate.	['gloves', 'woman', 'screwdriver', 'egg']	['knife', 'plate']

"""

"""
len2 实验结果:
指标	训练前	训练后	理想目标	结论
avg_pos_similarity	0.773	0.915↑	→1.0	✅ 成功提升肯定语义对齐
avg_neg_similarity	0.684	0.939↑	→1.0	✅ 否定语义对齐显著优化
avg_ortho_metric	0.987	0.715↓	→0.0	✅ 正交分离度接近且低于标签0.788
avg_l_pos_l_neg_similarity	0.788	0.788	-	标签固有相关性不变

配置信息：
config = {
    # 模型超参数
    'embed_dim': 512,  # CLIP文本特征的维度
    'hidden_dim': 256,  # 隐藏层维度
    
    # 训练超参数
    'lambda1': 1,  # 语义对齐损失权重
    'lambda2_max': 2,  # 正交对抗损失权重最大值  #TODO:待选择一个合适的值
    'ortho_eps': 0.5,  # 正交对抗损失阈值  #TODO:待选择一个合适的值，统计一下negbench数据集的正交度量平均值？？？
    'use_dynamic_weight': True,  # 是否使用动态权重
    'k': 5.0,  # 动态权重的斜率
    's0': 0.6,  # 动态权重的初始相似度
    
    # 训练常规超参数
    'epochs': 200,  # 训练轮数
    'batch_size': 32,  # 批次大小
    'lr': 1e-3,  # 学习率
    'early_stop_patience': 200,  # 早停耐心值
    'stage_switch_threshold': 0.02,  # 阶段切换阈值-越大-越早结束stage-1粗对齐，越早开启stage-2精细对齐
    
    # 数据集划分
    'train_rate': 0.8
}
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
        image = Clip_model.preprocess(image_path) # [num_classes, 3, 224, 224]
        image = image.to(Clip_model.device) # [num_classes, 3, 224, 224]
        image_features = Clip_model.visual(image) # [num_classes, embed_dim]
        return image_features.cpu().numpy()[0] # [embed_dim]
    

class CLIPGlassesLens(nn.Module):
    """
    CLIPGlassesLens: 一个轻量级模块，用于处理CLIP文本编码器的输出特征，
    生成否定内容和肯定内容的特征。
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
    
    def gate_reg_loss(self, h):
        """
        门控值正则化损失：门控值正则化，避免极端的门控值
        
        参数:
            - h: CLIP文本特征 [batch_size, embed_dim]
            
        返回:
            - gate_reg: 门控值正则化损失
        """     
        gates = self.gate_net(h)
        gate_reg = torch.mean((gates - 0.5) ** 2) * 0.01
        
        return gate_reg


class CLIPLensLoss(nn.Module):
    """
    CLIP-Lens的损失函数：三元组对比正交损失
    
    包含：
    1. 语义对齐损失：使生成的肯定/否定特征与目标特征对齐
    2. 正交对抗损失：强制肯定特征和否定特征在语义空间正交
    3. 动态权重调整：随训练进程增强正交约束
    """
    def __init__(self, config, lambda1, lambda2_max, total_steps):
        """
        初始化CLIP-Lens损失函数
        
        参数:
            config: 配置字典，包含超参数设置
        """
        super().__init__()
        
        self.lambda1 = lambda1 # 语义对齐损失权重
        self.lambda2_max = lambda2_max # 正交对抗损失权重
        self.total_steps = total_steps # 总训练步数
        
        self.ortho_eps = config['ortho_eps'] # 正交对抗损失阈值
        self.use_dynamic_weight = config['use_dynamic_weight'] # 是否使用动态权重(动态语义感知训练)
        self.k = config['k'] # 动态权重的斜率
        self.s0 = config['s0'] # 动态权重的初始相似度
        self.current_step = 0 # 当前训练步数
        
    def forward(self, h_pos, h_neg, l_pos, l_neg, h_original=None):
        """
        计算CLIP-Lens的损失
        
        参数:
            h_pos: 生成的肯定内容特征 [batch_size, embed_dim]
            h_neg: 生成的否定内容特征 [batch_size, embed_dim]
            l_pos: 目标肯定内容特征 [batch_size, embed_dim]
            l_neg: 目标否定内容特征 [batch_size, embed_dim]
            h_original: 原始CLIP文本特征 (用于难度感知加权，可选)
        """
        # 计算余弦相似度
        cos_h_pos_l_pos = F.cosine_similarity(h_pos, l_pos, dim=1, eps=1e-8)
        cos_h_neg_l_neg = F.cosine_similarity(h_neg, l_neg, dim=1, eps=1e-8)
        cos_h_pos_h_neg = F.cosine_similarity(h_pos, h_neg, dim=1, eps=1e-8)
        
        # 1. 语义对齐损失
        align_loss_pos = 1.0 - cos_h_pos_l_pos
        align_loss_neg = 1.0 - cos_h_neg_l_neg
        align_loss = 0.5 * (align_loss_pos + align_loss_neg)
        
        # 应用难度感知加权(如果启用)
        if self.use_dynamic_weight and h_original is not None:
            s = F.cosine_similarity(h_original, l_pos, dim=1, eps=1e-8) + F.cosine_similarity(h_original, l_neg, dim=1, eps=1e-8)
            w = 2.0 / (1.0 + torch.exp(-self.k * (s - self.s0)))
            align_loss = align_loss * w
        
        align_loss = torch.mean(align_loss)
        
        # 2. 正交对抗损失
        ortho_loss = torch.mean(torch.clamp(cos_h_pos_h_neg**2 - self.ortho_eps, min=0.0))
        
        # 3. 计算动态权重
        lambda2 = self.lambda2_max * (1.0 - torch.exp(torch.tensor(-5.0 * self.current_step / self.total_steps)))
        # print(f"Step {self.current_step}/{self.total_steps} - lambda2: {lambda2.item():.4f} - lambda1: {self.lambda1:.4f}")
        self.current_step += 1
        
        # 4. 总损失
        total_loss = self.lambda1 * align_loss + lambda2 * ortho_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'align_loss': align_loss.item(),
            'ortho_loss': ortho_loss.item(),
            'lambda2': lambda2.item()
        }
    
    def reset_step(self):
        """重置当前步数，用于新stage开始时"""
        self.current_step = 0


def load_dataset(file_path):
    """加载并预处理数据集"""
    df = pd.read_csv(file_path)
    
    dataset = []
    for _, row in df.iterrows():
        input_sentence = row['S_pn']
        pos_objects = eval(row['L_p'])
        neg_objects = eval(row['L_n'])
        
        # 构建目标文本
        pos_text = "A photo that includes " + ", ".join(pos_objects) if pos_objects else "A photo that includes nothing"
        neg_text = "A photo that includes " + ", ".join(neg_objects) if neg_objects else "A photo that includes nothing"
        
        dataset.append({
            'input_sentence': input_sentence,
            'pos_text': pos_text,
            'neg_text': neg_text,
            'pos_objects': pos_objects,
            'neg_objects': neg_objects
        })
    
    return dataset

def prepare_features(dataset, batch_size=32):
    """提取数据集中每个样本的CLIP文本特征"""
    features = []
    
    for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Extracting features"):
        batch = dataset[i:i+batch_size]
        
        # 提取输入句子特征
        input_sentences = [item['input_sentence'] for item in batch]
        input_features = []
        for sent in input_sentences:
            input_features.append(extract_sentence_features(sent))
        input_features = np.array(input_features)
        
        # 提取肯定和否定对象文本特征
        pos_texts = [item['pos_text'] for item in batch]
        neg_texts = [item['neg_text'] for item in batch]
        
        pos_features = []
        neg_features = []
        for pos_text, neg_text in zip(pos_texts, neg_texts):
            pos_features.append(extract_sentence_features(pos_text))
            neg_features.append(extract_sentence_features(neg_text))
        
        pos_features = np.array(pos_features)
        neg_features = np.array(neg_features)
        
        for j in range(len(batch)):
            features.append({
                'input_feature': input_features[j],
                'pos_feature': pos_features[j],
                'neg_feature': neg_features[j],
                'input_sentence': batch[j]['input_sentence'],
                'pos_text': batch[j]['pos_text'],
                'neg_text': batch[j]['neg_text']
            })
    
    return features

def train(model, features, config):
    """
    训练CLIP-Lens模型，实现两阶段训练策略
    
    参数:
        - model: CLIP-Lens模型
        - features: 训练数据集特征
        - config: 配置字典，包含超参数设置
    """
    # 超参数
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['lr']
    early_stop_patience = config['early_stop_patience']
    stage_switch_threshold = config['stage_switch_threshold']
    lambda1 = config['lambda1']
    lambda2_max = config['lambda2_max']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # 初始阶段只使用对齐损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CLIPLensLoss(lambda1=1.0, lambda2_max=0.0, total_steps=len(features)//batch_size*epochs, config=config)
    
    # 转换为张量
    input_features = torch.tensor(np.array([item['input_feature'] for item in features]), dtype=torch.float32)
    pos_features = torch.tensor(np.array([item['pos_feature'] for item in features]), dtype=torch.float32)
    neg_features = torch.tensor(np.array([item['neg_feature'] for item in features]), dtype=torch.float32)
    
    # 训练记录
    history = {'align_loss': [], 'ortho_loss': [], 'total_loss': []}
    best_loss = float('inf')
    patience_counter = 0
    stage = 1
    
    for epoch in range(epochs):
        total_loss = 0.0
        align_losses = []
        ortho_losses = []
        
        # 打乱数据
        indices = torch.randperm(len(features))
        input_features_shuffled = input_features[indices]
        pos_features_shuffled = pos_features[indices]
        neg_features_shuffled = neg_features[indices]
        
        # criterion.reset_step()
        
        # 批次训练
        for i in tqdm.tqdm(range(0, len(features), batch_size), desc=f"Epoch {epoch+1}/{epochs}, Stage {stage}"):
            batch_input = input_features_shuffled[i:i+batch_size].to(device)
            batch_pos_target = pos_features_shuffled[i:i+batch_size].to(device)
            batch_neg_target = neg_features_shuffled[i:i+batch_size].to(device)
            
            # 前向传播
            h_pos, h_neg = model(batch_input)
            
            # 计算损失
            loss, loss_dict = criterion(h_pos, h_neg, batch_pos_target, batch_neg_target, batch_input)
            
            # 添加门控正则损失
            gate_reg = model.gate_reg_loss(batch_input)
            loss += gate_reg
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item() * len(batch_input)
            align_losses.append(loss_dict['align_loss'])
            ortho_losses.append(loss_dict['ortho_loss'])
        
        # 计算平均损失
        epoch_loss = total_loss / len(features)
        epoch_align_loss = np.mean(align_losses)
        epoch_ortho_loss = np.mean(ortho_losses)
        
        history['total_loss'].append(epoch_loss)
        history['align_loss'].append(epoch_align_loss)
        history['ortho_loss'].append(epoch_ortho_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Align: {epoch_align_loss:.4f}, Ortho: {epoch_ortho_loss:.4f}")
        
        # 检查是否进入阶段2
        if stage == 1 and epoch >= 3:
            align_improvement = (history['align_loss'][-2] - history['align_loss'][-1]) / history['align_loss'][-2]
            if align_improvement < stage_switch_threshold:
                print(f"Switching to Stage 2: Align improvement ({align_improvement:.4f}) < threshold ({stage_switch_threshold})")
                stage = 2
                # 重置优化器学习率并更新损失函数
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = CLIPLensLoss(lambda1=lambda1, lambda2_max=lambda2_max, total_steps=len(features)//batch_size*(epochs-epoch), config=config)
        
        # 早停检查
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_clip_lens.pth'))
        else:
            patience_counter += 1 # 增加耐心计数器
            if patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # 加载最佳模型
    # model.load_state_dict(torch.load(os.path.join(current_dir, 'best_clip_lens.pth')))
    return model, history

@torch.no_grad()
def evaluate(model, features, batch_size=32):
    """评估CLIP-Lens模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    input_features = torch.tensor(np.array([item['input_feature'] for item in features]), dtype=torch.float32)
    pos_features = torch.tensor(np.array([item['pos_feature'] for item in features]), dtype=torch.float32)
    neg_features = torch.tensor(np.array([item['neg_feature'] for item in features]), dtype=torch.float32)
    
    # 评估指标
    pos_similarities = [] # 模型生成的肯定内容特征与目标肯定内容特征的相似度
    neg_similarities = [] # 模型生成的否定内容特征与目标否定内容特征的相似度
    ortho_metrics = [] # 模型生成的肯定内容特征与否定内容特征的正交度量
    
    pos_neg_similarities = [] # 模型生成的肯定内容特征与目标否定内容特征的相似度
    neg_pos_similarities = [] # 模型生成的否定内容特征与目标肯定内容特征的相似度
    l_pos_l_neg_similarities = [] # 目标肯定内容特征与目标否定内容特征的相似度
    
    
    for i in tqdm.tqdm(range(0, len(features), batch_size), desc="Evaluating"):
        batch_input = input_features[i:i+batch_size].to(device)
        batch_pos_target = pos_features[i:i+batch_size].to(device)
        batch_neg_target = neg_features[i:i+batch_size].to(device)
        
        h_pos, h_neg = model(batch_input)
        
        # 计算相似度
        cos_h_pos_l_pos = F.cosine_similarity(h_pos, batch_pos_target, dim=1)
        cos_h_neg_l_neg = F.cosine_similarity(h_neg, batch_neg_target, dim=1)
        cos_h_pos_h_neg = F.cosine_similarity(h_pos, h_neg, dim=1)
        
        cos_h_pos_l_neg = F.cosine_similarity(h_pos, batch_neg_target, dim=1)
        cos_h_neg_l_pos = F.cosine_similarity(h_neg, batch_pos_target, dim=1)
        cos_l_pos_l_neg = F.cosine_similarity(batch_pos_target, batch_neg_target, dim=1)
        
        pos_similarities.extend(cos_h_pos_l_pos.cpu().numpy())
        neg_similarities.extend(cos_h_neg_l_neg.cpu().numpy())
        ortho_metrics.extend(abs(cos_h_pos_h_neg.cpu().numpy()))
        
        pos_neg_similarities.extend(cos_h_pos_l_neg.cpu().numpy())
        neg_pos_similarities.extend(cos_h_neg_l_pos.cpu().numpy())
        l_pos_l_neg_similarities.extend(cos_l_pos_l_neg.cpu().numpy())
        
    
    metrics = {
        'avg_pos_similarity': np.mean(pos_similarities),
        'avg_neg_similarity': np.mean(neg_similarities),
        'avg_ortho_metric': np.mean(ortho_metrics),
        
        'avg_pos_neg_similarity': np.mean(pos_neg_similarities),
        'avg_neg_pos_similarity': np.mean(neg_pos_similarities),
        'avg_l_pos_l_neg_similarity': np.mean(l_pos_l_neg_similarities),
    }
    
    return metrics

@torch.no_grad()
def predict(model, sentence):
    """对新句子进行预测，返回肯定和否定内容特征"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    features = extract_sentence_features(sentence)
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    h_pos, h_neg = model(features_tensor)
    
    return h_pos.cpu().numpy()[0], h_neg.cpu().numpy()[0]

def visualize_examples(model, examples, top_k=5):
    """可视化模型预测效果"""
    print("="*50)
    print("CLIP-Lens Prediction Examples")
    print("="*50)
    
    candidates = ["dog", "cat", "chair", "table", "car", "bird", "fish", 
                 "laptop", "phone", "cup", "bottle", "book", "pen", "shoes", 
                 "hat", "clothes", "food", "tree", "flower", "person"]
    
    # 提取候选对象文本特征
    candidate_features = []
    for obj in candidates:
        text = f"A photo that includes {obj}."
        feature = extract_sentence_features(text)
        candidate_features.append(feature)
    candidate_features = np.array(candidate_features)
    
    for sentence in examples:
        print(f"\nInput: {sentence}")
        h_pos, h_neg = predict(model, sentence)
        
        # 计算相似度
        pos_similarities = np.dot(candidate_features, h_pos) / (np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(h_pos))
        neg_similarities = np.dot(candidate_features, h_neg) / (np.linalg.norm(candidate_features, axis=1) * np.linalg.norm(h_neg))
        
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

def load_model_example():
    """
    使用示例: 演示如何加载预训练的 CLIPGlassesLens 模型并进行预测
    """
    print("="*50)
    print("CLIPGlassesLens 模型加载示例")
    print("="*50)
    
    # 模型权重路径 - 请替换为实际路径
    weights_path = os.path.join(current_dir, 'best_clip_lens.pth')
    
    # 加载预训练模型
    model = load_clip_glasses_lens(weights_path)
    
    # 测试样例
    test_examples = [
        "In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves.",
        "On a wooden dining table amidst a quiet afternoon, you can see a bright gloves, a woman, a screwdriver, a delicious egg, and yet without a knife and no a plate."
    ]
    
    # 对测试样例进行预测
    for sentence in test_examples:
        print(f"\n输入句子: {sentence}")
        
        # 使用模型进行预测
        h_pos, h_neg = predict(model, sentence)
        
        # 预测结果可视化
        print(f"肯定内容特征范数: {np.linalg.norm(h_pos):.4f}")
        print(f"否定内容特征范数: {np.linalg.norm(h_neg):.4f}")
        
        # 计算特征之间的余弦相似度
        cos_sim = np.dot(h_pos, h_neg) / (np.linalg.norm(h_pos) * np.linalg.norm(h_neg))
        print(f"肯定与否定特征余弦相似度: {cos_sim:.4f}")
        
        print("-"*50)
    
    return model


def main():
    
    # 配置字典      
    config = {
        # 模型超参数
        'embed_dim': 512,  # CLIP文本特征的维度
        'hidden_dim': 256,  # 隐藏层维度
        
        # 训练超参数
        'lambda1': 1,  # 语义对齐损失权重
        'lambda2_max': 2,  # 正交对抗损失权重最大值  
        'ortho_eps': 0.5,  # 正交对抗损失阈值
        'use_dynamic_weight': True,  # 是否使用动态权重
        'k': 5.0,  # 动态权重的斜率
        's0': 0.6,  # 动态权重的初始相似度
        
        # 训练常规超参数
        'epochs': 200,  # 训练轮数
        'batch_size': 32,  # 批次大小
        'lr': 1e-3,  # 学习率
        'early_stop_patience': 200,  # 早停耐心值
        'stage_switch_threshold': 0.02,  # 阶段切换阈值
        
        # 数据集划分
        'train_rate': 0.8
    }
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据集
    dataset_path = "/root/NP-CLIP/NegBench/data/MCQ_natural_polished.csv"
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # 划分训练集和验证集
    np.random.shuffle(dataset)
    train_size = int(config['train_rate'] * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # 提取特征
    print("Extracting features for training set...")
    train_features = prepare_features(train_dataset)
    print("Extracting features for validation set...")
    val_features = prepare_features(val_dataset)
    
    # 初始化模型
    clip_lens = CLIPGlassesLens(config)
    
    # 在训练前评估模型
    print("Evaluating model before training...")
    initial_metrics = evaluate(clip_lens, val_features)
    print(f"Initial evaluation metrics: {initial_metrics}")
    
    # 训练模型
    print("Starting training...")
    clip_lens, history = train(
        model=clip_lens,
        features=train_features,
        config=config,
    )
    
    # 评估模型
    print("Evaluating model...")
    metrics = evaluate(clip_lens, val_features)
    print(f"Evaluation metrics: {metrics}")
    
    # 保存模型
    torch.save(clip_lens.state_dict(), os.path.join(current_dir, 'final_clip_lens.pth'))
    print(f"Model saved to {os.path.join(current_dir, 'final_clip_lens.pth')}")
    
    # 测试示例
    test_examples = [
        "In a rustic cabin, an elegant bench sits in the corner, while with notable absence of a camera and no a gloves.",
        "On a wooden dining table amidst a quiet afternoon, you can see a bright gloves, a woman, a screwdriver, a delicious egg, and yet without a knife and no a plate.",
        "In the cozy living room, there's a comfortable sofa and a bookshelf, but no television or radio."
    ]
    
    visualize_examples(clip_lens, test_examples)
    
    # 演示如何加载预训练模型
    print("\n演示如何加载预训练的 CLIPGlassesLens 模型:")
    load_model_example()
    
    return clip_lens, history, metrics


if __name__ == "__main__":
    # main() # 训练
    load_model_example() # 加载使用
