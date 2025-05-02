import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # 将输出重定向到log.txt文件
set_random_seed(3407)  # 设置随机种子
from Lens import CLIPGlassesLens
from Frame import CLIPGlassesFrame
from NegDetector import NegationDetector
from McqDataset import McqDataset, evaluate_model_mcq
from RetrievalDataset_gtneg import RetrievalNegGtDataset, evaluate_model_retrieval_withGTNeg
from RetrievalDataset import RetrievalDataset, evaluate_model_retrieval, retrieval_collate_fn
from CCNegDataset_gtneg import CCNegGtDataset, evaluate_model_CCNeg_etrieval_withGTNeg
from CLSDataset import CLSDataset, evaluate_model_CLS
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F


class Glasses(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.device = cfg['device']
        self.lens = CLIPGlassesLens.load_model(cfg['Lens'])
        self.frame = CLIPGlassesFrame.load_model(cfg['Frame'])
        self.negDetector = NegationDetector.load_model(cfg['NegationDetector']) # 轻量级否定分类器 | 1:包含否定 0:肯定
        self.neg_thr = cfg['NegationDetector']['neg_thr'] # 否定阈值
        self.dtype = cfg['dtype']
       
        # 冻结negDetector
        for param in self.negDetector.parameters():
            param.requires_grad = False
    
    def forward(self, I, h, level_h_list, l_neg=None):
        """
        参数:
            - I: 图像特征 [N_imgs=B, D]
            - h: 最后一层特征 [N_caps=B*num_options, D]
            - level_h_list: 各层特征列表 [N_caps=B*num_options, L, D]
            - l_neg: 被否定对象的文本特征 [N_caps=B*num_options, D] | 当为None时，使用lens预测
        返回:
            - scores_T2I: 文本->图像的分数 [N_caps, N_imgs=B]
            - scores_I2T: 图像->文本的分数 [N_imgs=B, N_caps]
        """
        # 否定检测
        with torch.no_grad():
            neg_mask = self.negDetector(h).squeeze(-1) > self.neg_thr # 否定阈值
        
        # Lens        
        if l_neg is None:
            h_neg = self.lens(h, level_h_list)
        else:
            h_neg = l_neg # 测试直接使用GT的h_neg
        assert I.size(0) == h_neg.size(0) == h.size(0), f"frame要求图片应该和文本一对一对应"
        
        # Frame
        # scores_T2I = self.frame(I, h, h_neg)
        scores_T2I = self.frame(I, h, h_neg, neg_mask=neg_mask) # 增加了neg_mask
        scores_I2T = scores_T2I.T
        
        return scores_T2I, scores_I2T
    
    def calc_losses(self, scores_T2I, scores_I2T, caption_to_img):
        caption_to_img = torch.tensor(caption_to_img, device=self.device, dtype=torch.long)
        # Text→Image contrastive loss -> 可简化为CrossEntropyLoss
        loss_txt2img = F.cross_entropy(scores_T2I, caption_to_img)
        # Image→Text contrastive loss -> 由于一个图片可能对应多个 caption，因此需要对每个图像的所有 caption 特征进行 softmax
        exp_sim = scores_T2I.exp() # [N_caps, B]
        all_exp = exp_sim.sum(dim=0) # [B]
        # mask[c, i] = 1 -> caption c 属于图 i
        mask = torch.zeros_like(exp_sim) # [N_caps, B]
        mask[torch.arange(exp_sim.size(0), device=self.device), caption_to_img] = 1
        pos_exp = (exp_sim * mask).sum(dim=0) # [B] # 正样本对应的logits
        loss_img2txt = - (pos_exp / all_exp).log().mean() # softmax
        contrastive_loss = 0.5*(loss_txt2img + loss_img2txt)
        total_loss = contrastive_loss
        return total_loss, {'contrastive_loss': contrastive_loss.item()}
    
    def calc_ccneg_losses(scores_T2Ip, scores_Ip2T):
        """
        - Ip: 正样本图像 [N]
        - hp: 正样本文本 [N]
        - hn: 难负样本文本 [N]
        
        参数：
            scores_T2Ip : [2N, N] 文本到正图像相似度矩阵（前N为hp，后N为hn）
            scores_Ip2T : [N, 2N] 正图像到文本相似度矩阵
        """
        batch_size = scores_Ip2T.size(0)
        device = scores_Ip2T.device
        
        # 构造标签映射
        # 图像到文本：每个图像i的正样本为hp_i（索引i）
        labels_I2T = torch.arange(batch_size, device=device)
        
        # 文本到图像：前N个hp的正样本为图像i，后N个hn无正样本（设为-1）
        labels_T2I = torch.cat([
            torch.arange(batch_size, device=device),
            -torch.ones(batch_size, device=device)  # hn无对应图像
        ])
        
        # 计算图像到文本损失
        loss_I2T = F.cross_entropy(scores_Ip2T, labels_I2T)
        
        # 计算文本到图像损失（仅考虑前N个hp）
        valid_mask = (labels_T2I != -1)
        valid_scores = scores_T2Ip[valid_mask]
        valid_labels = labels_T2I[valid_mask].long()
        if valid_labels.numel() > 0:
            loss_T2I = F.cross_entropy(valid_scores, valid_labels)
        else:
            loss_T2I = torch.tensor(0.0, device=device)
        
        # 加权平均
        total_loss = (loss_I2T + loss_T2I) / 2
        
        return total_loss, {
            'loss_I2T': loss_I2T.item(),
            'loss_T2I': loss_T2I.item()
        }
        
    def calc_ccneg_4_losses(scores_Tpn2Ip, scores_Ip2Tpn, scores_In2Tpn, scores_Tpn2In):
        """
        - Ip: 正样本图像 [N]
        - In: 负样本图像 [N]（与hn一一对应）
        - hp: 正样本文本 [N]
        - hn: 难负样本文本 [N]
        
        参数：
            scores_Tpn2Ip : [2N, N] 文本到正图像相似度矩阵（前N为hp，后N为hn）
            scores_Ip2Tpn : [N, 2N] 正图像到文本相似度矩阵
            scores_In2Tpn : [2N, N] 文本到负图像相似度矩阵（前N为hp，后N为hn）
            scores_Tpn2In : [N, 2N] 负图像到文本相似度矩阵
            caption_to_img : [N] 每个caption对应图像索引（此处应为0~N-1）
        """
        batch_size = scores_Ip2Tpn.size(0)
        device = scores_Ip2Tpn.device
        
        # ========== 正图像-正文本对 ==========
        # 正图像Ip匹配正文本hp
        labels_Ip = torch.arange(batch_size, device=device)
        
        # Ip2T损失：每个Ip应匹配对应的hp
        loss_Ip2Tpn = F.cross_entropy(scores_Ip2Tpn, labels_Ip) # Ip应匹配对应hp
        
        # T2Ip损失：hp应匹配对应Ip（排除hn）
        hp_scores_Tpn2Ip = scores_Tpn2Ip[:batch_size]  # 前N行hp
        loss_Tpn2Ip = F.cross_entropy(hp_scores_Tpn2Ip, labels_Ip) # hp应匹配对应Ip

        # ========== 负图像-难负文本对 ==========
        # 反转文本顺序：hn在前，hp在后（适配In匹配hn）
        hn_scores_In2Tpn = scores_In2Tpn[:, batch_size:]  # In匹配hn的分数 [N, N]
        hp_scores_In2Tpn = scores_In2Tpn[:, :batch_size]   # In匹配hp的分数 [N, N]
        scores_In2Tnp = torch.cat([hn_scores_In2Tpn, hp_scores_In2Tpn], dim=1)  # [N, 2N] 反转Tp和Tn的顺序
        
        # In2T损失：每个In应匹配对应的hn
        loss_In2Tnp = F.cross_entropy(scores_In2Tnp, labels_Ip) # In应匹配对应hn
        
        # TODO: xjh新增，待实验结果判断是否添加
        # # T2In损失：hn应匹配对应In
        loss_Tpn2In = torch.tensor(0.0, device=device)
        # hn_scores_Tpn2In = scores_Tpn2In[batch_size:]  # 后N行hn
        # loss_Tpn2In = F.cross_entropy(hn_scores_Tpn2In, labels_Ip) # hn应匹配对应In
        
        # ========== 综合损失 ==========
        total_loss = (loss_Ip2Tpn + loss_Tpn2Ip + loss_In2Tnp + loss_Tpn2In)/4
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'loss_Ip2T': loss_Ip2Tpn.item(),
            'loss_T2Ip': loss_Tpn2Ip.item(),
            'loss_In2T': loss_In2Tnp.item(),
            'loss_T2In': loss_Tpn2In.item(),
        }
        
        
    @staticmethod
    def load_model(cfg):
        """
        加载模型
        参数:
            - cfg: 配置文件
            - model_path: 模型路径
        返回:
            - model: 加载的模型
        """
        model = Glasses(cfg)
        
        # 导入NegationDetector的权重
        print(f"正在加载 NegationDetector 模型权重: {cfg['NegationDetector']['model_path']}")
        model.negDetector.load_state_dict(torch.load(cfg['NegationDetector']['model_path'], weights_only=True))
        
        # 导入Lens和Frame的权重
        if 'pretrain' in cfg.keys() and cfg['pretrain'] and cfg['model_path'] is not None:
            print(f"训练：正在加载预训练 Glasses 模型权重: {cfg['model_path']}, 将覆盖 Lens 和 Frame 的权重，不覆盖 NegationDetector 的权重")
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        if 'test' in cfg.keys() and cfg['test'] is True and cfg['model_path'] is not None:
            print(f"测试：正在加载被测试 Glasses 模型权重: {cfg['model_path']}, 将覆盖 Lens 和 Frame 的权重，不覆盖 NegationDetector 的权重")
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        
        return model


def train_COCORetr_with_gtneg(cfg, model:Glasses, with_gt_neg=True):   
    """
    训练Glasses模型 | 代理任务: Retrieval with gtneg
    
    参数:
        - cfg: 配置文件
    """
    # 读取配置
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False # 如果cfg中没有clip_grad，则默认不裁剪
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    train_rate, val_rate, test_rate = cfg['RetrievalWithGtNeg']['split']

    # 创建数据集和数据加载器
    dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    print(f">>> train_rate, val_rate, test_rate: {train_rate}, {val_rate}, {test_rate}")
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 优化器
    if cfg.get('only_train_moudle', None) == 'lens':
        print("只训练lens模型")
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        print("只训练frame模型")
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else: # 训练所有模块
        print("训练Glasses所有模块")
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 梯度监控钩子
    print("注册梯度监控钩子")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"梯度 {name} 范数: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    
    # 训练前测试
    evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    
    # Training loop
    best_recall5 = 0
    patience_counter = 0
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        losses = {'contrastive_loss': 0}
              
        # 遍历每一个batch
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            h = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_h = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device) # 肯定文本特征 [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # 被否定对象的文本特征 [batch_size, embed_dim]
            I = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            image_ids = batch['img_id'].to(device) # 图像ID [batch_size]
            
            unique_img_ids, remapped_ids = torch.unique(image_ids, sorted=True, return_inverse=True)
            caption_to_img = remapped_ids.cpu().numpy()
            
            # Forward pass
            if with_gt_neg is True:
                scores_T2I, scores_I2T = model(I, h, level_h, l_neg) # 使用GT的h_neg
            else:
                scores_T2I, scores_I2T = model(I, h, level_h) # 使用lens预测的h_neg
            
            # 将 scores_T2I 根据 caption_to_img 从 [N_caps, N_imgs] 还原为 [N_caps, N_imgs]
            cti = torch.tensor(caption_to_img, dtype=torch.long, device=device)  # [N_caps]
            unique_vals = torch.unique(cti, sorted=True)
            first_idx = []
            for val in unique_vals:
                idx = (cti == val).nonzero(as_tuple=True)[0][0]
                first_idx.append(idx)
            first_idx = torch.stack(first_idx, dim=0)  # [N_imgs]
            scores_T2I = scores_T2I[:, first_idx]  # [N_caps, N_imgs]
            scores_I2T = scores_T2I.t()
            
            # Compute loss
            loss, loss_dict = model.calc_losses(scores_T2I, scores_I2T, caption_to_img)
            epoch_loss += loss.item()
            losses['contrastive_loss'] += loss_dict['contrastive_loss']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} contrastive_loss: {losses['contrastive_loss']/batch_count:.4f}")
        scheduler.step()    
        
        # validation
        val_recall5 = evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)['mean'][5] # mean-recall@5 
        
        # Save best model
        if val_recall5 > best_recall5:
            best_recall5 = val_recall5
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with recall@5: {best_recall5}")
        else: # 早停
            patience_counter += 1 # 增加耐心计数器
            print(f"💔recall5 drop from {best_recall5:.4f} to {val_recall5:.4f}, cur patience_counter add to {patience_counter}")
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break    
        # Save checkpoint
        if epoch % 5 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'recall5': val_recall5
            }
            torch.save(checkpoint, os.path.join(current_dir, f"checkpoint_epoch_{epoch}.pth"))
        
        print(f"Training completed. Best validation recall5: {best_recall5:.4f}")
    
    return model


def train_CCNeg_with_gtneg(cfg, model:Glasses, with_gt_neg=True):   
    """
    CCNeg数据集:
        
    def __getitem__(self, idx):
        return {
            'Ip': self.data[idx]['I'], # 正样本图像特征 [embed_dim]
            'In': self.data[top1_index]['I'] # 负样本图像特征 [embed_dim] 
            'hp': self.data[idx]['hp'], # 肯定文本特征 [embed_dim]
            'hn': self.data[idx]['hn'], # 加了否定词的干扰错误文本特征 [embed_dim]
            'level_hp_list': self.data[idx]['level_hp_list'], # (每层)否定文本特征列表 [num_layers, embed_dim]
            'level_hn_list': self.data[idx]['level_hn_list'], # (每层)加了否定词的干扰错误文本特征列表 [num_layers, embed_dim]
            'l_pos': self.data[idx]['l_pos'], # 肯定文本特征 [embed_dim]
            'l_neg': self.data[idx]['l_neg'], # 加了否定词的干扰错误文本特征 [embed_dim]
            'neg_obj': self.data[idx]['neg_obj'], # 否定对象的文本特征 [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'], # 图像路径
            'img_id': self.data[idx]['img_id'], # 图像ID
        }

    训练Glasses模型 | 代理任务: CCNeg with gtneg
    
    参数:
        - cfg: 配置文件
        - model: Glasses模型
        - with_gt_neg: 是否使用GT的h_neg
    """
    # 读取配置
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience']
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    train_rate, val_rate, test_rate = cfg['CCNegGtDataset']['split']

    # 创建数据集和数据加载器
    dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    print(f">>> train_rate, val_rate, test_rate: {train_rate}, {val_rate}, {test_rate}")
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 优化器
    if cfg.get('only_train_moudle', None) == 'lens':
        print("只训练lens模型")
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        print("只训练frame模型")
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else: # 训练所有模块
        print("训练Glasses所有模块")
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 梯度监控钩子
    print("注册梯度监控钩子")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"梯度 {name} 范数: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    
    # 训练前测试
    evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    
    # Training loop
    best_recall5 = 0
    patience_counter = 0
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        losses = {'contrastive_loss': 0}
                
        # 遍历每一个batch
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            Ip = batch['I'].to(device)  # 图像特征 [batch_size, embed_dim]
            In = batch['In'].to(device)  # 负样本图像特征 [batch_size, embed_dim]
            hp = batch['hp'].to(device)  # 肯定文本特征 [batch_size, embed_dim]
            hn = batch['hn'].to(device)  # 加了否定词的干扰错误文本特征 [batch_size, embed_dim]
            level_hp_list = batch['level_hp_list'].to(device)  # 肯定文本特征列表 [batch_size, num_layers, embed_dim]
            level_hn_list = batch['level_hn_list'].to(device)  # 否定文本特征列表 [batch_size, num_layers, embed_dim]
            neg_obj = batch['neg_obj'].to(device)  # 否定对象的文本特征 [batch_size, embed_dim]
            img_id = batch['img_id'].to(device)  # 图像ID [batch_size]
            
            batch_size = Ip.size(0)
            
            # Forward pass for both positive and negative text features
            if with_gt_neg is True:
                _, scores_Ip2Tp = model(Ip, hp, level_hp_list, neg_obj) # I2T [num_images=N, num_texts=N]
                _, scores_Ip2Tn = model(Ip, hn, level_hn_list, neg_obj) 
                _, scores_In2Tp = model(In, hp, level_hp_list, neg_obj) # I2T [num_images=N, num_texts=N]
                _, scores_In2Tn = model(In, hn, level_hn_list, neg_obj)
            else:
                _, scores_Ip2Tp = model(Ip, hp, level_hp_list) # I2T [num_images=N, num_texts=N]
                _, scores_Ip2Tn = model(Ip, hn, level_hn_list)
                _, scores_In2Tp = model(In, hp, level_hp_list) # I2T [num_images=N, num_texts=N]
                _, scores_In2Tn = model(In, hn, level_hn_list)
            
            scores_Ip2T = torch.cat([scores_Ip2Tp, scores_Ip2Tn], dim=1) # I2T [num_images=N, num_texts=2N]
            scores_T2Ip = scores_Ip2T.t() # T2I [num_texts=2N, num_images=N]
            scores_In2T = torch.cat([scores_In2Tp, scores_In2Tn], dim=1) # I2T [num_images=N, num_texts=2N]
            scores_T2In = scores_In2T.t() # T2I [num_texts=2N, num_images=N]
            
            # 每个图片对应一个hp（正样本文本）和 hn（难例负样本，通过否定化hp实现，无可匹配图像），batch内其余图片的hp和hn为普通负样本
            loss, loss_dict = model.calc_ccneg_losses(scores_T2Ip, scores_Ip2T, scores_In2T, scores_T2In)
            
            epoch_loss += loss.item()
            losses['contrastive_loss'] += loss_dict['contrastive_loss']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} contrastive_loss: {losses['contrastive_loss']/batch_count:.4f}")
        scheduler.step()    
        
        # validation
        val_results = evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
        val_recall5 = val_results['mean'][5]  # mean-recall@5 
        
        # Save best model
        if val_recall5 > best_recall5:
            best_recall5 = val_recall5
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with recall@5: {best_recall5}")
        else:  # 早停
            patience_counter += 1  # 增加耐心计数器
            print(f"💔recall5 drop from {best_recall5:.4f} to {val_recall5:.4f}, cur patience_counter add to {patience_counter}")
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break    
        
        # Save checkpoint
        if epoch % 5 == 0 or epoch == epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'recall5': val_recall5
            }
            torch.save(checkpoint, os.path.join(current_dir, f"checkpoint_epoch_{epoch}.pth"))
        
    print(f"Training completed. Best validation recall5: {best_recall5:.4f}")
    
    return model

if __name__ == "__main__":
    # Example usagerue
    cfg = {
        # -----训练参数-----
        'epochs': 30,
        'batch_size': 64,
        # 'lr': 5e-3, # 57.47%
        'lr': 1e-5, # r@5: 57.73%
        # 'lr': 1e-4, # r@5: 58.82%
        # 'lr': 10, # r@5: 57.91% - 36.37%
        # 'lr': 1e-5, # r@5: 57.33
        'num_workers': 4,
        'early_stop_patience': 5, # Early stopping patience
        'device': 'cuda',
        'dtype': torch.float32,
        'save_path': 'best_clip_Glasses.pth', # 训练得到的模型权重保存路径
        'pretrain': False, # 是否使用预训练Glasses
        
        # -----模型参数-----
        'Lens': {
            'device': 'cuda',
            'dtype': torch.float32,
            'num_heads': 4,
            'dropout': 0.1,
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/v1/best_clip_lens_9922.pth' # Lens的预训练权重
        },
        'Frame': {
            'device': 'cuda',
            'dtype': torch.float32,
            'lambda_0': 1, # 基础惩罚强度
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_Frame_mse_v1869.pth' # Frame的预训练权重
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/best_clip_Frame.pth' # Frame的预训练权重
        },
        'NegationDetector': {
            'device': 'cuda',
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth', # NegationDetector的预训练权重
            'neg_thr': 0.5, # 否定阈值(大于该值则为否定) 例如：全否定: -1.0, 全肯定: 1.0
        },
        
        # -----数据参数-----
        'Mcq': {
            'batch_size': 64,
            'num_workers': 4,
            'num_options': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',
            # 'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv', # 35.90%
            'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv',  # 41.66%
        },
        'Retrieval': {
            'batch_size': 64,
            'num_workers': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
            'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
        },
        'RetrievalWithGtNeg': { # h_neg直接作为GT给出，只训练和测试Frame
            'batch_size': 64,
            'num_workers': 4, 
            'split': [0.9, 0.1, 0.0],  # train, val, test split
            'pos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
            'negpos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
            'dtype': torch.float32, 
        },
        'CCNegGtDataset': {
            'batch_size': 64,
            'num_workers': 4,
            'split': [0.9, 0.1, 0.0],  # train, val, test split
            'csv_path': '/root/NP-CLIP/NegBench/data/ccneg_converted.csv',
            'dtype': torch.float32, 
            
        },
        'ClsEvalDataset': {
            'csv_path': '/root/NP-CLIP/NegBench/data/CLS_Imagenet/imagenet_train.csv',
            'batch_size': 64,
            'num_workers': 4,
        }
        
    }

    # # 一阶段训练：使用gtneg代替lens输出，单独训练Frame模型，不训练Lens模型 -- Recall@5: 99.71%
    # cfg['lr'] = 1e-4
    # cfg['neg_thr'] = -1
    # cfg['epochs'] = 10
    # model = Glasses.load_model(cfg)
    # model = train_Retrieval_with_gtneg(cfg, model, with_gt_neg=True) # 一阶段: 训练Glasses模型 | 代理任务: Retrieval with gtneg
    
    # # 二阶段训练：使用GT_neg作为监督单独训练lens, 在lens.py中完成
    
    # # 三阶段训练：联合训练lens和Frame模型，进行适配 -- Recall@5: val: 75.97% full: 82.24%  -- MCQ: 35.90%
    # cfg['pretrain'] = True
    # cfg['lr'] = 1e-3
    # cfg['model_path'] = 'best_clip_Glasses.pth' # 一阶段预模型权重路径
    # cfg['neg_thr'] = -1
    # cfg['clip_grad'] = True # 梯度裁剪
    # model = Glasses.load_model(cfg)
    # model.lens = CLIPGlassesLens.load_model(cfg['Lens']) # 加载lens模型的预训练权重
    # model = train_Retrieval_with_gtneg(cfg, model, with_gt_neg=False) # 二阶段: 联合lens训练Glasses模型 | 代理任务: Retrieval
    
    # 测试模型通用配置
    cfg['test_raw_clip'] = True
    cfg['test'] = True
    # cfg['model_path'] = 'weights/v1/best_clip_Glasses_7597_8224_3590(after_joint).pth' # 测试模型权重路径
    cfg['model_path'] = 'weights/v2/best_clip_Glasses.pth'
    # cfg['model_path'] = 'best_clip_Glasses.pth'
    cfg['Lens']['model_path'], cfg['Frame']['model_path'] = None, None # 不覆盖joint训练后的Glasses
    cfg['NegationDetector']['model_path'] = '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth'
    
    # 测试Imagenet传统分类能力保留程度
    cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/imagenet_val.csv' # ours:52.40% CLIP:53.87%
    test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    
    # 测试Imagenet1K传统分类能力保留程度
        
    # # 测试caltech101传统分类能力保留程度
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/caltech101.csv' # ours:90.54% clip:90.74%
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
        
    # # 测试CIFAR-100传统分类能力保留程度
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/cifar100.csv' # ours:38.50% clip:37.04%
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
        
    # # 测试CIFAR-10传统分类能力保留程度
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/cifar10.csv' # ours:71.03% clip:71.08%
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    
    # # 测试 CC-Neg
    # test_ccneg_dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    # test_ccneg_dataloader = torch.utils.data.DataLoader(test_ccneg_dataset, batch_size=cfg['CCNegGtDataset']['batch_size'], shuffle=False, num_workers=cfg['CCNegGtDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CCNeg_etrieval_withGTNeg(None, test_ccneg_dataloader, test_raw_clip=True, with_gt_neg=False)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CCNeg_etrieval_withGTNeg(model, test_ccneg_dataloader, test_raw_clip=False, with_gt_neg=False) # 使用lens预测的h_neg
    
    # # 测试 Retrieval with gtbeg
    # test_retrieval_dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_retrieval_withGTNeg(None, test_retrieval_dataloader, test_raw_clip=True, with_gt_neg=False)
    # else:
    #     model = Glasses.load_model(cfg)
    #     # evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False, with_gt_neg=True) # 使用GT的h_neg
    #     evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False, with_gt_neg=False) # 使用lens预测的h_neg
        
    # # 测试 MCQ VOC 
    # cfg['Mcq']['test_dataset_path'] = '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv'
    # test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
        
    # # 测试 MCQ COCO
    # cfg['Mcq']['test_dataset_path'] = '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv'
    # test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
    
    
    print("==============配置项===============")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k1, v1 in v.items():
                print(f"  {k1}: {v1}")
        else:
            print(f"{k}: {v}")
    print("===================================")