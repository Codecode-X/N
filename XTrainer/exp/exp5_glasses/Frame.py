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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from GlassesDataset import GlassesDataset
from torch.utils.data import DataLoader

class CLIPGlassesFrame(nn.Module):
    def __init__(self, cfg, embed_dim=512, hidden_dim=2048):
        super().__init__()
        self.cfg = cfg
        self.lambda_0 = cfg['lambda_0']
        self.register_buffer('logit_scale', Clip_model.logit_scale.detach())
        
        # 深度跨模态交互模块（3层Transformer）
        self.cross_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.3,
                activation='gelu',
                batch_first=True,
                layer_norm_eps=1e-6 # 增加数值稳定性
            ),
            num_layers=3
        )
        
        # 多模态特征融合网络（带门控残差连接）
        self.feature_fusion = nn.Sequential(
            *[ResidualBlock(embed_dim*3, hidden_dim) for _ in range(2)],
            nn.Linear(embed_dim*3, embed_dim),  # 新增维度压缩层
            nn.LayerNorm(embed_dim)
        )
        
        # 动态lambda生成器（交叉注意力机制）
        self.lambda_generator = nn.ModuleDict({
            'cross_attn': nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=0.2,
                batch_first=True
            ),
            'gate_controller': nn.Sequential(
                nn.Linear(2*embed_dim, 1),
                nn.Sigmoid()
            )
        })
        
        # 自适应残差系数
        self.alpha = nn.Parameter(torch.ones(2)*0.1)  # 多阶段残差
        
        # 初始化适配
        self._init_weights()

    def _init_weights(self):
        # Transformer初始化
        for p in self.cross_transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 特征融合网络初始化
        for module in self.feature_fusion:
            if isinstance(module, ResidualBlock):
                nn.init.kaiming_normal_(module.fc1.weight, mode='fan_in')
                nn.init.zeros_(module.fc2.weight)
                nn.init.xavier_normal_(module.gate.weight)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, I, h, h_neg, neg_mask=None):
        """
        参数：
            - I: 图像特征 [batch_size, embed_dim]
            - h: CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            - h_neg: 被否定对象的文本特征 [batch_size, embed_dim]
            - neg_mask: 是否有否定对象mask [batch_size] | 1:有否定对象 | 0: 无否定对象
        
        返回：
            - scores: CLIPGlassesFrame 计算得到的 h2I 匹配得分 [N_caps, N_imgs]
        """
        # 特征归一化
        I_norm = F.normalize(I, p=2, dim=-1)
        h_norm = F.normalize(h, p=2, dim=-1)
        h_neg_norm = F.normalize(h_neg, p=2, dim=-1) + 1e-8
        
        # 深度跨模态交互
        cross_features = self.cross_transformer(
            torch.cat([h_norm.unsqueeze(1), I_norm.unsqueeze(1)], dim=1)
        )
        h_attn = self.alpha[0]*h_norm + cross_features[:,0]
        
        # 多模态特征融合
        fused_feature = self.feature_fusion(
            torch.cat([
                h_attn, 
                h_neg_norm, 
                h_attn * h_neg_norm  # 新增交互特征
            ], dim=-1)
        )
        
        # 动态lambda生成（交叉注意力机制）
        attn_output, _ = self.lambda_generator['cross_attn'](
            query=h_attn.unsqueeze(1),
            key=h_neg_norm.unsqueeze(1),
            value=h_neg_norm.unsqueeze(1)
        )
        gate_input = torch.cat([h_attn, attn_output.squeeze(1)], dim=-1)
        lambda_base = self.lambda_generator['gate_controller'](gate_input)
        lambda_dynamic = torch.sigmoid(self.lambda_0 * lambda_base) # 动态lambda生成器，限制在[0,1]之间
        
        # 保持CLIP基础能力的自适应匹配
        with torch.amp.autocast('cuda', enabled=True):
            logit_scale = self.logit_scale.exp()
            
            # 原CLIP预测的h和I的匹配分数 
            scores_base = logit_scale * (h_norm @ I_norm.t())
            
            # 增强匹配路径
            enhanced_feature = self.alpha[0]*h_norm + self.alpha[1]*fused_feature
            scores_enhanced = logit_scale * (enhanced_feature @ I_norm.t())
            
            # 否定感知调整
            scores_N2I = logit_scale * (h_neg_norm @ I_norm.t())
            adjusted_scores = scores_enhanced - lambda_dynamic * scores_N2I
            
            # 条件混合
            if neg_mask is not None:
                neg_mask = neg_mask.to(scores_base.dtype)
                scores = torch.where(
                    neg_mask.bool().view(-1,1),  # 将mask转换为[B,1]用于行广播
                    adjusted_scores + scores_base.detach(),  # True时使用修正分数
                    scores_base  # False时使用原始分数
                )
            else:
                scores = adjusted_scores + scores_base.detach() # 无mask时保持原有逻辑 | scores_base.detach() 防止梯度回传到原始CLIP模型
        
        return scores.float()
    
    @staticmethod
    def load_model(cfg):
        """
        Load the trained CLIPGlassesFrame model from a checkpoint
        
        参数:
            - cfg: 配置参数
            - model_path: 模型路径
            
        返回:
            - model: 加载的模型
        """
        model = CLIPGlassesFrame(cfg)
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"正在加载 CLIPGlassesFrame 模型权重: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=False))
        model = model.to(cfg['device'])
        model.eval()
        return model

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gate = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.clamp(x, min=-5.0, max=5.0)  # 梯度截断
        x = self.fc2(x)
        
        # 门控残差连接
        gate = torch.sigmoid(self.gate(residual))
        return self.layer_norm(gate * x + (1 - gate) * residual)
    
    
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
        
    @staticmethod
    def load_model(cfg):
        """
        Load the trained CLIPGlassesFrame model from a checkpoint
        
        参数:
            - cfg: 配置参数
            - model_path: 模型路径
            
        返回:
            - model: 加载的模型
        """
        model = CLIPGlassesFrame(cfg)
        if 'model_path' in cfg.keys() and cfg['model_path'] is not None:
            print(f"正在加载 CLIPGlassesFrame 模型权重: {cfg['model_path']}")
            model.load_state_dict(torch.load(cfg['model_path'], weights_only=False))
        model = model.to(cfg['device'])
        model.eval()
        return model
    
 
def train(cfg, model:CLIPGlassesFrame, Lens_model=None, device='cuda'):
    """
    Train the CLIPGlassesFrame model
    
    参数:
        - cfg: 配置参数
        - model: CLIPGlassesFrame模型
        - Lens_model: Lens_model=None 表示使用GT neg_obj进行训练，否则使用冻结的Lens模型预测的neg_obj进行训练
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
    patience_counter = 0 # Early stopping counter
    for epoch in range(epochs):
        model.train()
        best_loss = float('inf')
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
            if Lens_model is None: # 使用GT neg_obj 训练 MLP
                scores = model(I, h, h_neg=l_neg) # 使用GT l_neg 训练 MLP
            else: # 使用冻结的Lens模型预测的neg_obj进行训练
                with torch.no_grad():
                    h_neg = Lens_model(h, level_h_list)
                scores = model(I, h, h_neg=h_neg) # 使用Lens模型预测的neg_obj进行训练

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
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
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


if __name__ == "__main__":
    # 配置参数
    cfg = {
        # -----模型参数-----
        'dtype': torch.float32,
        'device': 'cuda',
        'lambda_0': 0.1, # 基础惩罚强度
        
        'model_path': os.path.join(current_dir, 'weights/best_clip_Frame_mse_v1869.pth'), # 预训练模型权重的路径
        'save_path': os.path.join(current_dir, 'best_clip_Frame.pth'), # 训练得到的模型权重保存路径
        
        'rank_loss_weight': 0.5, # 排名损失权重
        'margin': 0.5, # 排名损失的margin
        
        'Lens': {
            'device': 'cuda',
            'dtype': torch.float32,
            'num_heads': 4,
            'dropout': 0.1,
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_lens_9832_0027.pth' # Lens的预训练权重
        },
        
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
    
    # 加载当前Frame模型预训练权重
    if cfg['model_path'] is not None:
        print(f"正在加载 CLIPGlassesFrame 模型权重: {cfg['model_path']}")
        model.load_state_dict(torch.load(cfg['model_path'], weights_only=True))
    
    # 加载冻结的预训练的Lens模型
    from Lens import CLIPGlassesLens
    lens_model = CLIPGlassesLens.load_model(cfg['Lens'])
    for param in lens_model.parameters():
        param.requires_grad = False
    lens_model.eval()
    
    # 训练模型
    trained_model = train(cfg, model) # 直接使用 GT neg_obj 进行训练
    # trained_model = train(cfg, model, lens_model) # 使用冻结的Lens模型预测的 neg_obj 进行训练

    # model = CLIPGlassesFrame.load_model(cfg)
    # model.eval()
    # model = model.to('cuda')
    # data_loader = DataLoader(GlassesDataset(cfg), batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], drop_last=True)
    # evaluate(cfg, model, data_loader)
    