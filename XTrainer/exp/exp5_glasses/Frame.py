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
    """
    CLIPGlassesFrame: 
        输入：
            - CLIP图像编码器的输出图像特征I。
            - CLIP输出的文本特征h。
            - Lens输出的否定内容文本特征h_neg。
        输出:
            - 文本和图像的匹配度
        场景：
            - Retrieval任务
    """
    def __init__(self, cfg, embed_dim=512, hidden_dim=128):
        """
        初始化CLIPGlassesFrame模块
        
        Args:
            - embed_dim: 嵌入维度(CLIP特征维度)
            - hidden_dim: MLP隐藏层维度
            - lambda_0: 基础惩罚强度
        """
        super().__init__()
        self.cfg = cfg
        self.lambda_0 = cfg['lambda_0']  # 基础惩罚强度
        self.logit_scale = Clip_model.logit_scale.detach() # 直接使用CLIP模型的训练好的logit_scale
        self.confidence_mlp = nn.Sequential( # 用于动态惩罚权重的MLP
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, I, h, h_neg):
        """
        计算图像和文本特征的匹配得分
        
        Args:
            I: 图像特征 [N_imgs, embed_dim]
            h: 原内容文本特征 [N_caps, embed_dim]
            h_neg: 否定内容文本特征 [N_caps, embed_dim]
            
        Returns:
            scores: 匹配得分 [N_caps, N_imgs]
        """
        # 计算动态惩罚权重
        lambda_dynamic = self.lambda_0 * torch.sigmoid(self.confidence_mlp(h_neg)) # [N_caps, 1]
        
        # 标准化
        I = I / I.norm(dim=-1, keepdim=True) # [N_imgs, embed_dim]
        h = h / h.norm(dim=-1, keepdim=True) # [N_caps, embed_dim]
        h_neg = h_neg / h_neg.norm(dim=-1, keepdim=True) # [N_caps, embed_dim]
        
        # 计算标准化差分匹配得分
        logit_scale = self.logit_scale.exp()
        scores_H2I = logit_scale * h @ I.t() # [N_caps, N_imgs]
        scores_N2I = logit_scale * h_neg @ I.t() # [N_caps, N_imgs]
        scores = scores_H2I - lambda_dynamic * scores_N2I
        return scores
    
    def calc_losses(self, scores, I, l_pos):
        """
        蒸馏训练 用于计算动态惩罚权重 的 MLP:

        参数:
            - cfg: 配置参数
            - scores: CLIPGlassesFrame 计算得到的 h2I 匹配得分 [N_caps, N_imgs]
            - I: 图像特征 [N_imgs, embed_dim]
            - l_pos: 肯定内容文本特征 [N_caps, embed_dim]
            
        返回:
            - total_loss: 总损失
            - loss_dict: 各个分项损失的字典 | MSE损失
        """
        
        # calc similarity
        I = I / I.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
        l_pos = l_pos / l_pos.norm(dim=-1, keepdim=True) # [batch_size, embed_dim]
        logit_scale = self.logit_scale.exp()
        scores_gt = logit_scale * l_pos @ I.t()
        
        # mse
        loss_mse = F.mse_loss(scores, scores_gt, reduction='none')
        
        return loss_mse.mean(), {
            'mse_loss': loss_mse.mean().item(),
        }
 
    
def train(cfg, model:CLIPGlassesFrame, device='cuda'):
    """
    Train the CLIPGlassesLens model
    
    参数:
        - cfg: 配置参数
        - model: CLIPGlassesLens模型
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
        
    dataset = GlassesDataset(cfg) # Clip_model, lens_model 用于预加载数据过程中的特征提取
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
        best_loss = float('inf')
        patience_counter = 0 # Early stopping counter
        total_loss = 0
        losses = {'mse_loss': 0}
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            h = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_h_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device) # 肯定文本特征 [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # 被否定对象的文本特征 [batch_size, embed_dim]
            I = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            
            # Forward pass
            scores = model(I, h, h_neg=l_neg) # 使用GT l_neg 训练 MLP
            
            # Compute loss
            loss, loss_dict = model.calc_losses(scores, I, l_pos)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            losses['mse_loss'] += loss_dict['mse_loss']
            # losses['pos2negobj_sim_loss'] += loss_dict['pos2negobj_sim_loss']
        
        scheduler.step()
        
        # Print epoch summary
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {total_loss/batch_count:.4f} mse_loss: {losses['mse_loss']/batch_count:.4f}")
        
        # Validation
        batch_loss = evaluate(cfg, model, val_loader)
        
        # 早停
        if batch_loss < best_loss:
            best_loss = batch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, 'best_clip_lens.pth'))
        else:
            print(f"💔loss improve from {best_loss:.4f} to {batch_loss:.4f}, cur patience_counter add to {patience_counter}")
            patience_counter += 1 # 增加耐心计数器
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
    return model


def evaluate(cfg, model, data_loader):
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
    device = next(model.parameters()).device
    total_loss = 0
    losses = {'mse_loss': 0}
    
    with torch.no_grad():  # No need to track gradients during evaluation
        for batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            h = batch['h'].to(device)
            level_h_list = batch['level_h_list'].to(device)
            l_pos = batch['l_pos'].to(device)
            l_neg = batch['neg_obj'].to(device)  # Negative object features
            I = batch['I'].to(device)
            
            # Forward pass
            scores = model(I, h, h_neg=l_neg)
            
            # Compute loss
            loss, loss_dict = model.calc_losses(scores, I, l_pos)
            
            # Track metrics
            total_loss += loss.item()
            losses['mse_loss'] += loss_dict['mse_loss']
    
    batch_count = len(data_loader)
    avg_loss = total_loss / batch_count
    print(f"Validation - Loss: {avg_loss:.4f}, MSE Loss: {losses['mse_loss']/batch_count:.4f}")
    
    return avg_loss


if __name__ == "__main__":
    # 配置参数
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        'lambda_0': 0.1, # 基础惩罚强度
        
        # -----训练参数-----
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'split': [0.9, 0.1, 0.0], # train val test
        'num_workers': 4,
        'early_stop_patience': 5,
        
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
