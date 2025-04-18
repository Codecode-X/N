"""
实验结果:
指标	训练前	训练后	理想目标	结论
avg_pos_similarity	0.773	0.926↑	→1.0	✅ 成功提升肯定语义对齐
avg_neg_similarity	0.684	0.934↑	→1.0	✅ 否定语义对齐显著优化
avg_ortho_metric	0.987	0.820↓	→0.0	⚠️ 正交分离不足，需重点改进
avg_pos_neg_similarity	0.685	0.822↑	→0.0	❌ 反向相关性恶化
avg_neg_pos_similarity	0.772	0.822↑	→0.0	❌ 交叉干扰增强
avg_l_pos_l_neg_similarity	0.788	0.788	-	标签固有相关性不变

"""

"""
设计并训练一个轻量级模块CLIP-Lens，通过处理CLIP的文本编码器的输出文本特征，来生成否定内容和肯定内容的文本特征。

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

config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"

Clip_model = build_clip_model(config_path=config_path) # 加载CLIP模型

def extract_sentence_features(sentence:str):
    """提取单个句子的CLIP文本特征"""
    with torch.no_grad():  # 关闭梯度计算
        tokenized_text = tokenize(sentence) # [num_classes, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes, context_length]
        text_features = Clip_model.encode_text(tokenized_text) # [num_classes, embed_dim]
        return text_features.cpu().numpy()[0] # [embed_dim]
    


"""
### CLIP-Lens模块 设计思路

---

#### **1. 模块结构设计：双路正交投影网络**

**核心思想**：通过正交约束的投影空间，强制分离肯定/否定语义，同时保留CLIP的语义空间结构。

- **结构设计**：

  - **共享特征分解层**：**轻量级MLP**将CLIP特征 `h` 分解为两个**正交基向量 `u`（肯定相关）和 `v`（否定相关）**。

  - **动态门控融合**：基于输入特征 `h` 的语义强度，动态融合正交基向量生成最终特征：
    $$
    h_{pos} = \alpha \cdot u + (1-\alpha) \cdot h_{original}
    $$

    $$
    h_{neg} = \beta \cdot v + (1-\beta) \cdot h_{original}
    $$

    其中 $\alpha$, $\beta$ 由门控网络生成，用于控制语义修正强度。

    $$
    [\alpha, \beta] = \sigma(\text{MLP}(h))
    $$

    - **正则**：门控 MLP 输出后，用一个**小正则**（如 $\ell_2$ 惩罚）**抑制 $\alpha,\beta$ 过接近 0 或 1**，避免极端值使分解过度或失效。

- **创新点**：

  - **正交基约束**：强制 `u` 和 `v` 在特征空间正交，物理上分离两种语义。
  - **残差连接**：残差连接设计保留CLIP原有优势，在非否定场景下自动退化为原始特征

#### **2. 损失函数设计：三元组对比正交损失**

**核心思想**：通过对比学习对齐目标特征，同时引入正交约束和语义纯度约束。

- **损失组成**：

  1. **语义对齐损失**：
     $$
     \mathcal{L}_{align} = \frac{1}{2} \left[ 1 - \cos(h_{pos}, l_{pos}) + 1 - \cos(h_{neg}, l_{neg}) \right]
     $$


    2. **正交对抗损失**：
       $$
       \mathcal{L}_{ortho} = \max(0, \cos(h_{pos}, h_{neg})^2 - \epsilon)
       $$

  > 设置阈值 $\epsilon$ (如0.1)，允许微弱相关性但强制主要成分正交。

- **总损失**：

  $$
  \mathcal{L}_{total} = \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{ortho}
  $$

  $$
  \lambda_2(t) = \lambda_{2}^{max} \cdot (1 - e^{-5t/T})
  $$

  > *T* 为总训练步数，初期 $λ_2$ 较低，后期增强正交约束。

  > **建议**：**监控两项损失的梯度范数比**，若不平衡可考虑动态平衡（如 GradNorm）或自适应 $\lambda_1,\lambda_2$。

---

#### **3. 动态语义感知训练**

**核心思想**：根据样本难度自适应调整学习目标，缓解简单/困难样本的不平衡。

- **实现方法**：

  - **难度感知加权**：对每个样本计算初始相似度 $s = \cos(h, l_{pos}) + \cos(h, l_{neg})$，定义权重：
    $$
    w = \frac{2}{1 + e^{-k(s - s_0)}} \quad (k>0)
    $$


    - 对困难样本（初始相似度低）赋予更高权重。

  > **建议**：初期实验中验证该加权是否真的带来收敛速度或性能提升；若效果不明显，可将其作为可选模块。

"""

import torch.nn as nn
import torch.nn.functional as F

class CLIPLens(nn.Module):
    """
    CLIP-Lens: 一个轻量级模块，用于处理CLIP文本编码器的输出特征，
    生成否定内容和肯定内容的特征。
    """
    def __init__(self, embed_dim, hidden_dim=512):
        """
        初始化CLIP-Lens模块。
        
        参数:
            embed_dim: CLIP文本特征的维度
            hidden_dim: MLP隐藏层的维度
        """
        super().__init__()
        
        # 共享特征分解网络 - 输出正交基向量
        self.decomp_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 2)  # 输出正交基向量 u 和 v
        )
        
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
        
        # 获取正交基向量
        decomp = self.decomp_net(h)
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
    def __init__(self, 
                 lambda1=1.0, 
                 lambda2_max=1.0, 
                 ortho_eps=0.1,
                 use_dynamic_weight=False,
                 k=1.0, 
                 s0=0.0,
                 total_steps=1000):
        super().__init__()
        self.lambda1 = lambda1 # 语义对齐损失权重
        self.lambda2_max = lambda2_max # 正交对抗损失权重
        self.ortho_eps = ortho_eps # 正交对抗损失阈值
        self.use_dynamic_weight = use_dynamic_weight # 是否使用动态权重(动态语义感知训练)
        self.k = k # 动态权重的斜率
        self.s0 = s0 # 动态权重的初始相似度
        self.total_steps = total_steps # 总训练步数
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
        """重置当前步数，用于新epoch开始时"""
        self.current_step = 0


"""
### **训练流程**

1. **两阶段训练策略**：

   - **阶段1**（粗对齐）：仅使用 $\mathcal{L}_{align}$ 建立初步关联
   - **阶段2**（精细化）：逐步引入 $\mathcal{L}_{ortho}$ 和 $\mathcal{L}_{purity}$
   - **自动阶段切换**：当 $\mathcal{L}_{align}$ 连续3个epoch下降<1%时，进入阶段2。
   - **学习率热重启**：阶段切换时重置优化器学习率至初始值，避免陷入局部最优

2. **动态阈值调整**：

   - 根据训练进度自动调整正交约束阈值 $\epsilon$：

     $$
     \epsilon(t) = \epsilon_{min} + (\epsilon_{max} - \epsilon_{min}) \cdot e^{-t/\tau}
     $$


     - 初始宽松（$\epsilon_{max}=0.5$），逐步收紧至目标值（$\epsilon_{min}=0.1$）
     - **阈值 $\epsilon$** 的动态调整策略很有意义，但要注意指数衰减超参 $\tau$ 的选取，否则可能过早收紧导致训练崩。

   > **建议**：
   >
   > - 将阶段切换和 $\epsilon(t)$ 曲线可视化（如 TensorBoard），及时调整 $\tau$、下降阈值和 LR 重启策略。

"""
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

def train(model, features, epochs=20, batch_size=32, lr=2e-4, early_stop_patience=10, stage_switch_threshold=0.01):
    """训练CLIP-Lens模型，实现两阶段训练策略"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # 初始阶段只使用对齐损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = CLIPLensLoss(lambda1=1.0, lambda2_max=0.0, total_steps=len(features)//batch_size*epochs)
    
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
        
        criterion.reset_step()
        
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
                criterion = CLIPLensLoss(lambda1=1.0, lambda2_max=1.0, total_steps=len(features)//batch_size*(epochs-epoch))
        
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
    model.load_state_dict(torch.load(os.path.join(current_dir, 'best_clip_lens.pth')))
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

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 加载数据集
    dataset_path = "/root/NP-CLIP/NegBench/data/MCQ_natural_polished.csv"
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # 划分训练集和验证集
    np.random.shuffle(dataset)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # 提取特征
    print("Extracting features for training set...")
    train_features = prepare_features(train_dataset)
    print("Extracting features for validation set...")
    val_features = prepare_features(val_dataset)
    
    # 初始化模型
    embed_dim = train_features[0]['input_feature'].shape[0]
    clip_lens = CLIPLens(embed_dim=embed_dim, hidden_dim=512)
    
    # 在训练前评估模型
    print("Evaluating model before training...")
    initial_metrics = evaluate(clip_lens, val_features)
    print(f"Initial evaluation metrics: {initial_metrics}")
    
    # 训练模型
    print("Starting training...")
    clip_lens, history = train(
        model=clip_lens,
        features=train_features,
        epochs=200,
        batch_size=32,
        lr=2e-4
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
    
    return clip_lens, history, metrics

if __name__ == "__main__":
    main()
