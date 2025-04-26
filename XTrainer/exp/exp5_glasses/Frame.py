import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import Clip_model
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # 将输出重定向到log.txt文件
set_random_seed(3407)  # 设置随机种子
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from GlassesDataset import GlassesDataset
from torch.utils.data import DataLoader

class CLIPGlassesFrame(nn.Module):
    def __init__(self, cfg, embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.cfg = cfg
        self.lambda_0 = cfg['lambda_0']
        self.register_buffer('logit_scale', Clip_model.logit_scale.detach())
        
        # 跨模态交互模块
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            # dropout=0.5, # 1.8826
            dropout=0.7, # 1.8698
            # dropout=0.8, # 1.8905
            # dropout=0.9, # 1.9377
            batch_first=True
        )
        
        # 增强的特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(embed_dim*3, hidden_dim),
            nn.GELU(),
            # nn.Dropout(0.5), # 1.8826
            nn.Dropout(0.7), # 1.8698
            # nn.Dropout(0.8), # 1.8905
            # nn.Dropout(0.9), # 1.9377
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 动态lambda生成器（双通道结构）
        self.lambda_generator = nn.ModuleDict({
            'semantic_branch': nn.Sequential(
                nn.Linear(embed_dim*2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)),
            'syntactic_branch': nn.Sequential(
                nn.Linear(embed_dim*2, hidden_dim//2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim//2)),
            'fusion': nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim//2, 1),
                nn.Sigmoid())
        })
        
        # 残差连接参数
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        # 跨模态注意力初始化
        nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0.0)
        
        # 特征融合网络初始化
        nn.init.kaiming_normal_(self.feature_fusion[0].weight, mode='fan_in')
        nn.init.zeros_(self.feature_fusion[-2].weight)

    def forward(self, I, h, h_neg):
        # 特征归一化
        I_norm = F.normalize(I, p=2, dim=-1)
        h_norm = F.normalize(h, p=2, dim=-1)
        h_neg_norm = F.normalize(h_neg, p=2, dim=-1)
        
        # 跨模态注意力交互
        attn_output, _ = self.cross_attn(
            query=h_norm.unsqueeze(1),
            key=I_norm.unsqueeze(1),
            value=I_norm.unsqueeze(1)
        )
        h_attn = h_norm + self.alpha * attn_output.squeeze(1)
        
        # 多层次特征融合
        fused_feature = self.feature_fusion(
            torch.cat([h_attn, h_neg_norm, h_attn-h_neg_norm], dim=-1)
        )
        
        # 双通道lambda生成
        semantic_feat = self.lambda_generator['semantic_branch'](torch.cat([fused_feature, h_neg_norm], dim=-1))
        syntactic_feat = self.lambda_generator['syntactic_branch'](torch.cat([h_norm, h_neg_norm], dim=-1))
        lambda_base = self.lambda_generator['fusion'](torch.cat([semantic_feat, syntactic_feat], dim=-1))
        lambda_dynamic = self.lambda_0 * lambda_base
        
        # 稳定化得分计算
        with torch.amp.autocast('cuda', enabled=True):
            scores_H2I = self.logit_scale.exp() * (h_attn @ I_norm.t())
            scores_N2I = self.logit_scale.exp() * (h_neg_norm @ I_norm.t())
            scores = scores_H2I - lambda_dynamic * scores_N2I
            
        return scores.float()  # 确保输出精度
    
    def calc_losses(self, scores, I, l_pos, img_ids, h=None):
        """
        蒸馏训练 动态惩罚权重MLP:

        参数:
            - cfg: 配置参数
            - scores: CLIPGlassesFrame 计算得到的 h2I 匹配得分 [N_caps, N_imgs]
            - I: 图像特征 [N_imgs, embed_dim]
            - l_pos: 肯定内容文本特征 [N_caps, embed_dim]
            - img_ids: 图像ID [N_imgs]
            - h: CLIP文本编码器最后一层的输出文本特征(EOS特征) [N_caps, embed_dim]
            
        返回:
            - total_loss: 总损失
            - loss_dict: 各个分项损失的字典 | MSE损失
        """
        
        # calc similarity
        I = I / I.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
        l_pos = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
        logit_scale = self.logit_scale.exp()
        scores_gt = logit_scale * l_pos @ I.t() # [batch_size, batch_size]
        
        if h is not None:
            h = h / h.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
            raw_scores = logit_scale * h @ I.t()
            # mse
            print("="*50)
            print(f"G scores: {scores_gt.diag()}")
            print(f"P scores: {scores.diag()}")
            print(f"R scores: {raw_scores.diag()}")
        
        mse_loss = F.mse_loss(scores.diag(), scores_gt.diag(), reduction='none').mean()
        
        # 多caption感知排名损失
        margin = cfg['margin']  # 可配置参数
        # 构建图像分组掩码
        _, inverse_indices = torch.unique(img_ids, return_inverse=True)
        group_mask = inverse_indices.unsqueeze(1) == inverse_indices.unsqueeze(0)  # [B,B]
        # 计算正样本得分（同图像所有caption的最大得分）
        pos_scores = torch.where(group_mask, scores, -torch.inf).max(dim=1)[0]  # [B]
        # 计算负样本得分（不同图像所有caption的最小得分）
        neg_scores = torch.where(~group_mask, scores, torch.inf).min(dim=1)[0]  # [B]
        # 排名损失计算
        rank_loss = F.relu(neg_scores - pos_scores + margin).mean() * cfg['rank_loss_weight']
        total_loss = mse_loss + rank_loss
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'rank_loss': rank_loss.item(),
        }
    
 
def train(cfg, model:CLIPGlassesFrame, device='cuda'):
    """
    Train the CLIPGlassesFrame model
    
    参数:
        - cfg: 配置参数
        - model: CLIPGlassesFrame模型
        - device: 设备类型（'cuda'或'cpu'）
        
    返回:
        - model: 训练后的模型
    """
            
    if cfg:
        epochs = cfg['epochs']
        batch_size = cfg['batch_size']
        lr = cfg['lr']
        weight_decay = cfg['weight_decay']
        train_size, val_size, test_size = cfg['split']
        num_workers = cfg['num_workers']
        early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
       
    dataset = GlassesDataset(cfg) # Clip_model, Frame_model 用于预加载数据过程中的特征提取
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    # scheduler = OneCycleLR(
    #     optimizer, 
    #     max_lr=lr, 
    #     total_steps=epochs*len(train_loader),
    #     pct_start=0.3
    # )

    # Training loop
    for epoch in range(epochs):
        model.train()
        best_loss = float('inf')
        patience_counter = 0 # Early stopping counter
        total_loss = 0
        # losses = {'mse_loss': 0}
        losses = {'mse_loss': 0, 'rank_loss': 0}
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            h = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_h_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device) # 肯定文本特征 [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # 被否定对象的文本特征 [batch_size, embed_dim]
            I = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            img_ids = batch['img_id'].to(device) # 图像ID [batch_size]
            
            # Forward pass
            scores = model(I, h, h_neg=l_neg) # 使用GT l_neg 训练 MLP
            
            # Compute loss
            loss, loss_dict = model.calc_losses(scores, I, l_pos, img_ids)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['mse_loss'] += loss_dict['mse_loss']
            losses['rank_loss'] += loss_dict['rank_loss']
        
        # scheduler.step()
        
        # Print epoch summary
        batch_count = len(train_loader)
        # print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f} mse_loss: {losses['mse_loss']/batch_count:.4f}")
        print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f} mse_loss: {losses['mse_loss']/batch_count:.4f} rank_loss: {losses['rank_loss']/batch_count:.4f}")
        
        # Validation
        if epoch % 10 == 0: # 每隔1个epoch进行一次验证
            batch_loss = evaluate(cfg, model, val_loader, vis=True)
        else:
            batch_loss = evaluate(cfg, model, val_loader)
        # 早停
        if batch_loss < best_loss:
            best_loss = batch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_clip_Frame.pth'))
        else:
            patience_counter += 1 # 增加耐心计数器
            print(f"💔loss improve from {best_loss:.4f} to {batch_loss:.4f}, cur patience_counter add to {patience_counter}")
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    return model


def evaluate(cfg, model:CLIPGlassesFrame, data_loader, vis=False, device='cuda'):
    """
    Evaluate the CLIPGlassesFrame model on the validation set
    
    参数:
        - cfg: 配置参数
        - model: CLIPGlassesFrame模型
        - data_loader: 数据加载器
        
    返回:
        - avg_loss: 平均损失
    """
    model.eval()
    total_loss = 0
    losses = {'mse_loss': 0, 'rank_loss': 0}
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            l_neg = batch['neg_obj'].to(device)  # Negative object features
            I = batch['I'].to(device)
            img_ids = batch['img_id'].to(device) # 图像ID [batch_size]
            
            # Forward pass
            scores = model(I, h, h_neg=l_neg)
            
            # Compute loss
            if vis:
                loss, loss_dict = model.calc_losses(scores, I, l_pos, img_ids, h)
            else:
                loss, loss_dict = model.calc_losses(scores, I, l_pos, img_ids)
            
            # Track metrics
            total_loss += loss.item()
            losses['mse_loss'] += loss_dict['mse_loss']
            losses['rank_loss'] += loss_dict['rank_loss']
    
    batch_count = len(data_loader)
    avg_loss = total_loss / batch_count
    # print(f"Validation - Loss: {avg_loss:.4f}, MSE Loss: {losses['mse_loss']/batch_count:.4f}")
    print(f"Validation - Loss: {avg_loss:.4f}, mse_loss: {losses['mse_loss']/batch_count:.4f} rank_loss: {losses['rank_loss']/batch_count:.4f}")
    
    return avg_loss

# 加载模型并测试
def load_model(cfg, model_path):
    """
    Load the trained CLIPGlassesFrame model from a checkpoint
    
    参数:
        - cfg: 配置参数
        - model_path: 模型路径
        
    返回:
        - model: 加载的模型
    """
    model = CLIPGlassesFrame(cfg)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == "__main__":
    # 配置参数
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        'lambda_0': 0.1, # 基础惩罚强度
        'margin': 0.5,
        'rank_loss_weight': 0.5,
        
        # -----训练参数-----
        # 'epochs': 10, # 1.8698
        'epochs': 20,
        # 'batch_size': 32,
        'batch_size': 8,
        # 'lr': 1e-3,
        'lr': 5e-5, # 1.8698
        # 'lr': 1e-5, # 1.8931
        'weight_decay': 1e-4,
        'split': [0.9, 0.1, 0.0], # train val test
        # 'num_workers': 64,
        'num_workers': 4,
        'early_stop_patience': 3,
        
        # -----数据参数-----
        'pos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
        'negpos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
        'split': [0.9, 0.1, 0.0],  # train, val, test split
        'num_workers': 4,
    }
    
    # 创建模型
    model = CLIPGlassesFrame(cfg)
    
    # 训练模型
    trained_model = train(cfg, model)

    # model = load_model(cfg, os.path.join(current_dir, 'best_clip_Frame.pth'))
    # model.eval()
    # model = model.to('cuda')
    # data_loader = DataLoader(GlassesDataset(cfg), batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=True)
    # evaluate(cfg, model, data_loader)
    