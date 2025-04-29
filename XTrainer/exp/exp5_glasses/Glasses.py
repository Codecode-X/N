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
from McqDataset import McqDataset, evaluate_model_mcq
from RetrievalDataset_gtneg import RetrievalNegGtDataset, evaluate_model_retrieval_withGTNeg
from RetrievalDataset import RetrievalDataset, evaluate_model_retrieval, retrieval_collate_fn
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
        if l_neg is None:
            h_neg = self.lens(h, level_h_list)
        else:
            h_neg = l_neg # 测试直接使用GT的h_neg
        assert I.size(0) == h_neg.size(0) == h.size(0), f"frame要求图片应该和文本一对一对应"
        scores_T2I = self.frame(I, h, h_neg)
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
        if 'pretrain' in cfg.keys() and cfg['pretrain'] and cfg['model_path'] is not None:
            print(f"训练：正在加载预训练 Glasses 模型权重: {cfg['model_path']}, 将覆盖 Lens 和 Frame 的权重")
            model.load_state_dict(torch.load(os.path.join(current_dir, cfg['model_path']), weights_only=False))
        if 'test' in cfg.keys() and cfg['test'] is True and cfg['model_path'] is not None:
            print(f"测试：正在加载被测试 Glasses 模型权重: {cfg['model_path']}, 将覆盖 Lens 和 Frame 的权重")
            model.load_state_dict(torch.load(os.path.join(current_dir, cfg['model_path']), weights_only=False))
        return model
   
def train_Retrieval(cfg, model):   
    """
    训练Glasses模型 | 代理任务: Retrieval
    
    参数:
        - cfg: 配置文件
    """
    # 读取配置
    device = cfg['device']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    train_rate, val_rate, test_rate = cfg['Retrieval']['split']
    early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
    
    # 创建数据集和数据加载器
    dataset = RetrievalDataset(cfg['Retrieval']['train_dataset_path'])
    print(f">>> train_rate, val_rate, test_rate: {train_rate}, {val_rate}, {test_rate}")
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=retrieval_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=retrieval_collate_fn)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练前测试
    evaluate_model_retrieval(model, val_loader, test_raw_clip=False) 
    
    # Training loop
    best_recall5 = 0
    patience_counter = 0
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        losses = {'contrastive_loss': 0}
        
        # 遍历每一个batch
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            caption_feats = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_H_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device) # 肯定文本特征 [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # 被否定对象的文本特征 [batch_size, embed_dim]
            image_feats = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            image_ids = batch['img_id'].to(device) # 图像ID [batch_size]
                
            # prepare data
            all_image_feats = []
            all_text_feats  = [] # 原始CLIP最终输出的文本特征
            all_neg_feats = [] # 被否定对象的文本特征
            all_level_text_feats = [] # 原始CLIP每一层输出的文本特征列表
            caption_to_img  = [] # 记录 caption 属于哪张图
            
            B = image_feats.size(0)
            for batch_idx in range(B):
                # 把这张图 append 到 all_image_feats
                all_image_feats.append(image_feats[batch_idx].cpu())
                img_idx = len(all_image_feats)-1 # 这张图在 list 中的新索引
                # 把这张图的每条 caption 都收集起来，并记录它们对应 img_idx
                caps_b = caption_feats[batch_idx] # [num_caps_i, D]
                neg_b = l_neg[batch_idx] # [num_caps_i, D]
                levels_b = level_H_list[batch_idx] # [num_caps_i, L, D]
                for cap_f, lvl_f in zip(caps_b, levels_b):
                    all_text_feats.append(cap_f.cpu())  # [D]
                    all_neg_feats.append(neg_b.cpu()) # [D]
                    all_level_text_feats.append(lvl_f.cpu()) # [L, D]
                    caption_to_img.append(img_idx) # 记录 caption 属于哪张图
            
            # Stack 成大 tensor
            I = torch.stack(all_image_feats, dim=0).to(device)  # [N_imgs, D]
            h = torch.stack(all_text_feats, dim=0).to(device)  # [N_caps, D]
            l_neg = torch.stack(all_neg_feats, dim=0).to(device) # [N_caps, D]
            level_h = torch.stack(all_level_text_feats, dim=0).to(device) # [N_caps, L, D]
            N_imgs, N_caps = I.size(0), h.size(0)
            
            # 构造一对一的 I_rep，使其和 h/level_h 在 batch 维度上对齐
            idx = torch.tensor(caption_to_img, dtype=torch.long, device=device) # [N_caps]
            I_rep = I[idx]  # [N_caps, D]
            
            # Forward pass
            scores_T2I, scores_I2T = model(I_rep, h, level_h, l_neg)
            
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
            optimizer.step()
        
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} contrastive_loss: {losses['contrastive_loss']/batch_count:.4f}")
        scheduler.step()    
        
        # validation
        val_recall5 = evaluate_model_retrieval(model, val_loader, test_raw_clip=False)['mean'][5] # mean-recall@5 
        
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
        
if __name__ == "__main__":
    # Example usagerue
    cfg = {
        # -----训练参数-----
        'epochs': 30,
        'batch_size': 64,
        # 'lr': 5e-3, # 57.47%
        'lr': 1e-3, # r@5: 57.73%
        # 'lr': 1e-4, # r@5: 58.82%
        # 'lr': 10, # r@5: 57.91% - 36.37%
        # 'lr': 1e-5, # r@5: 57.33
        'num_workers': 4,
        'early_stop_patience': 10, # Early stopping patience
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
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_lens_9832_0027.pth' # Lens的预训练权重
        },
        'Frame': {
            'device': 'cuda',
            'dtype': torch.float32,
            'lambda_0': 0.1, # 基础惩罚强度
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_clip_Frame_mse_v1869.pth' # Frame的预训练权重
            # 'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/best_clip_Frame.pth' # Frame的预训练权重
        },

        # -----数据参数-----
        'Mcq': {
            'batch_size': 64,
            'num_workers': 4,
            'num_options': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',
            'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',
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
        }
    }
    print("==============配置项===============")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k1, v1 in v.items():
                print(f"  {k1}: {v1}")
        else:
            print(f"{k}: {v}")
    print("===================================")
    

    # # 训练模型
    # model = Glasses.load_model(cfg)
    # model = train_Retrieval(cfg, model)
    
    # 测试模型
    cfg['test_raw_clip'] = False, # 是否使用原始的CLIP模型进行测试
    cfg['test'] = True
    cfg['model_path'] = 'weights/best_clip_Glasses_5882.pth' # 测试模型权重路径
    # cfg['model_path'] = 'weights/best_clip_Glasses.pth' # 测试模型权重路径
    
    # Retrieval
    test_retrieval_dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'])
    if cfg['test_raw_clip'] is True:
        evaluate_model_retrieval_withGTNeg(None, test_retrieval_dataloader, test_raw_clip=True)
    else:
        model = Glasses.load_model(cfg)
        evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False)
        
    # test_retrieval_dataset = RetrievalDataset(cfg['Retrieval']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'], collate_fn=retrieval_collate_fn)
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_retrieval(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_retrieval(model, test_retrieval_dataloader, test_raw_clip=False)
        
    # # MCQ    
    # test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)