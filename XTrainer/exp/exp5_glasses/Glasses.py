import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from utils import setup_logger, set_random_seed
setup_logger(os.path.join(current_dir, "log.txt")) # Redirect output to log.txt file
set_random_seed(3407)  # Set random seed
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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F


class Glasses(nn.Module):
    def __init__(self, cfg):
        super().__init__() 
        self.device = cfg['device']
        self.lens = CLIPGlassesLens.load_model(cfg['Lens'])
        self.frame = CLIPGlassesFrame.load_model(cfg['Frame'])
        self.negDetector = NegationDetector.load_model(cfg['NegationDetector']) # Lightweight negation classifier | 1: contains negation, 0: affirmative
        self.neg_thr = cfg['NegationDetector']['neg_thr'] # Negation threshold
        self.dtype = cfg['dtype']
       
        # Freeze negDetector
        for param in self.negDetector.parameters():
            param.requires_grad = False
    
    def forward(self, I, h, level_h_list, l_neg=None, chunk_size=-1):
        """
        Args:
            - I: Image features [N_imgs=B, D]
            - h: Last layer features [N_caps=B*num_options, D]
            - level_h_list: Feature list for each layer [N_caps=B*num_options, L, D]
            - l_neg: Negated object text features [N_caps=B*num_options, D] | If None, use lens prediction
            - chunk_size: Chunk size | -1 means no chunking
        Returns:
            - scores_T2I: Text-to-image scores [N_caps, N_imgs=B]
            - scores_I2T: Image-to-text scores [N_imgs=B, N_caps]
        """
        # Negation detection
        with torch.no_grad():
            neg_mask = self.negDetector(h).squeeze(-1) > self.neg_thr # Negation threshold
        
        # Lens        
        if l_neg is None:
            h_neg = self.lens(h, level_h_list)
        else:
            h_neg = l_neg # Use GT h_neg during testing
        assert I.size(0) == h_neg.size(0) == h.size(0), f"Frame requires images and texts to be one-to-one"
        
        # Frame
        # scores_T2I = self.frame(I, h, h_neg)
        scores_T2I = self.frame(I, h, h_neg, neg_mask=neg_mask, chunk_size=chunk_size) # Added neg_mask | torch.Size([40000, num_imgs=64, num_caps=64])
        scores_I2T = scores_T2I.T
        
        return scores_T2I, scores_I2T
    
    def calc_losses(self, scores_T2I, scores_I2T, caption_to_img):
        caption_to_img = torch.tensor(caption_to_img, device=self.device, dtype=torch.long)
        # Text→Image contrastive loss -> Simplified as CrossEntropyLoss
        loss_txt2img = F.cross_entropy(scores_T2I, caption_to_img)
        # Image→Text contrastive loss -> Since one image may correspond to multiple captions, softmax is applied to all captions for each image
        exp_sim = scores_T2I.exp() # [N_caps, B]
        all_exp = exp_sim.sum(dim=0) # [B]
        # mask[c, i] = 1 -> caption c belongs to image i
        mask = torch.zeros_like(exp_sim) # [N_caps, B]
        mask[torch.arange(exp_sim.size(0), device=self.device), caption_to_img] = 1
        pos_exp = (exp_sim * mask).sum(dim=0) # [B] # Logits corresponding to positive samples
        loss_img2txt = - (pos_exp / all_exp).log().mean() # softmax
        contrastive_loss = 0.5*(loss_txt2img + loss_img2txt)
        total_loss = contrastive_loss
        return total_loss, {'contrastive_loss': contrastive_loss.item()}
    
    def calc_ccneg_losses(self, scores_T2Ip, scores_Ip2T):
        """
        - Ip: Positive sample images [N]
        - hp: Positive sample texts [N]
        - hn: Hard negative sample texts [N]
        
        Args:
            scores_T2Ip : [2N, N] Text-to-positive-image similarity matrix (first N are hp, last N are hn)
            scores_Ip2T : [N, 2N] Positive-image-to-text similarity matrix
        """
        batch_size = scores_Ip2T.size(0)
        device = scores_Ip2T.device
        
        # Construct label mapping
        # Image-to-text: Each image i's positive sample is hp_i (index i)
        labels_I2T = torch.arange(batch_size, device=device)
        
        # Text-to-image: First N hp's positive samples are image i, last N hn have no positive samples (set to -1)
        labels_T2I = torch.cat([
            torch.arange(batch_size, device=device),
            -torch.ones(batch_size, device=device)  # hn has no corresponding image
        ])
        
        # Compute image-to-text loss
        loss_I2T = F.cross_entropy(scores_Ip2T, labels_I2T)
        
        # Compute text-to-image loss (only consider the first N hp)
        valid_mask = (labels_T2I != -1)
        valid_scores = scores_T2Ip[valid_mask]
        valid_labels = labels_T2I[valid_mask].long()
        if valid_labels.numel() > 0:
            loss_T2I = F.cross_entropy(valid_scores, valid_labels)
        else:
            loss_T2I = torch.tensor(0.0, device=device)
        
        # Weighted average
        total_loss = (loss_I2T + loss_T2I) / 2
        
        return total_loss, {
            'loss_I2T': loss_I2T.item(),
            'loss_T2I': loss_T2I.item()
        }
        
    def calc_ccneg_4_losses(self, scores_Tpn2Ip, scores_Ip2Tpn, scores_In2Tpn, scores_Tpn2In):
        """
        - Ip: Positive sample images [N]
        - In: Negative sample images [N] (one-to-one with hn)
        - hp: Positive sample texts [N]
        - hn: Hard negative sample texts [N]
        
        Args:
            scores_Tpn2Ip : [2N, N] Text-to-positive-image similarity matrix (first N are hp, last N are hn)
            scores_Ip2Tpn : [N, 2N] Positive-image-to-text similarity matrix
            scores_In2Tpn : [2N, N] Text-to-negative-image similarity matrix (first N are hp, last N are hn)
            scores_Tpn2In : [N, 2N] Negative-image-to-text similarity matrix
            caption_to_img : [N] Each caption's corresponding image index (should be 0~N-1)
        """
        batch_size = scores_Ip2Tpn.size(0)
        device = scores_Ip2Tpn.device
        
        # ========== Positive image-positive text pair ==========
        # Positive image Ip matches positive text hp
        labels_Ip = torch.arange(batch_size, device=device)
        
        # Ip2T loss: Each Ip should match its corresponding hp
        loss_Ip2Tpn = F.cross_entropy(scores_Ip2Tpn, labels_Ip) # Ip should match corresponding hp
        
        # T2Ip loss: hp should match corresponding Ip (excluding hn)
        hp_scores_Tpn2Ip = scores_Tpn2Ip[:batch_size]  # First N rows hp
        loss_Tpn2Ip = F.cross_entropy(hp_scores_Tpn2Ip, labels_Ip) # hp should match corresponding Ip

        # ========== Negative image-hard negative text pair ==========
        # Reverse text order: hn first, hp last (adapt to In matching hn)
        hn_scores_In2Tpn = scores_In2Tpn[:, batch_size:]  # In matches hn scores [N, N]
        hp_scores_In2Tpn = scores_In2Tpn[:, :batch_size]   # In matches hp scores [N, N]
        scores_In2Tnp = torch.cat([hn_scores_In2Tpn, hp_scores_In2Tpn], dim=1)  # [N, 2N] Reverse Tp and Tn order
        
        # In2T loss: Each In should match its corresponding hn
        loss_In2Tnp = F.cross_entropy(scores_In2Tnp, labels_Ip) # In should match corresponding hn
        
        # TODO: xjh added, pending experimental results to determine whether to include
        # # T2In loss: hn should match corresponding In
        loss_Tpn2In = torch.tensor(0.0, device=device)
        # hn_scores_Tpn2In = scores_Tpn2In[batch_size:]  # Last N rows hn
        # loss_Tpn2In = F.cross_entropy(hn_scores_Tpn2In, labels_Ip) # hn should match corresponding In
        
        # ========== Combined loss ==========
        total_loss = (loss_Ip2Tpn + loss_Tpn2Ip + loss_In2Tnp + loss_Tpn2In)/4
        
        return total_loss, {
            'loss_Ip2Tpn': loss_Ip2Tpn.item(),
            'loss_Tpn2Ip': loss_Tpn2Ip.item(),
            'loss_In2Tnp': loss_In2Tnp.item(),
            'loss_Tpn2In': loss_Tpn2In.item(),
        }
        
        
    @staticmethod
    def load_model(cfg):
        """
        Load model
        Args:
            - cfg: Configuration file
            - model_path: Model path
        Returns:
            - model: Loaded model
        """
        model = Glasses(cfg)
        
        # Load weights for NegationDetector
        print(f"Loading NegationDetector model weights: {cfg['NegationDetector']['model_path']}")
        model.negDetector.load_state_dict(torch.load(cfg['NegationDetector']['model_path'], weights_only=True))
        
        # Load weights for Lens and Frame
        if 'pretrain' in cfg.keys() and cfg['pretrain'] and cfg['model_path'] is not None:
            print(f"Training: Loading pretrained Glasses model weights: {cfg['model_path']}, will overwrite Lens and Frame weights, but not NegationDetector weights")
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        if 'test' in cfg.keys() and cfg['test'] is True and cfg['model_path'] is not None:
            print(f"Testing: Loading tested Glasses model weights: {cfg['model_path']}, will overwrite Lens and Frame weights, but not NegationDetector weights")
            full_ckpt = torch.load(os.path.join(current_dir, cfg['model_path']), map_location='cpu', weights_only=False)
            filtered_ckpt = {k: v for k, v in full_ckpt.items() if not k.startswith("negDetector.")}
            model.load_state_dict(filtered_ckpt, strict=False)
        
        return model


def train_COCORetr_with_gtneg(cfg, model:Glasses, with_gt_neg=True):   
    """
    Train the Glasses model | Proxy task: Retrieval with gtneg
    
    Args:
        - cfg: Configuration file
    """
    # Read configuration
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False # If clip_grad is not in cfg, default to no clipping
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience'] # Early stopping patience
    lr = cfg['lr']
    num_workers = cfg['num_workers']
    train_rate, val_rate, test_rate = cfg['RetrievalWithGtNeg']['split']

    # Create dataset and data loaders
    dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    print(f">>> train_rate, val_rate, test_rate: {train_rate}, {val_rate}, {test_rate}")
    train_size = int(len(dataset) * train_rate)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Optimizer
    if cfg.get('only_train_moudle', None) == 'lens':
        print("Only training the lens module")
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        print("Only training the frame module")
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else: # Train all modules
        print("Training all modules of Glasses")
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Gradient monitoring hook
    print("Registering gradient monitoring hook")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"Gradient norm for {name}: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    
    # Pre-training evaluation
    evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    
    # Training loop
    best_recall5 = 0
    patience_counter = 0
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        losses = {'contrastive_loss': 0}
              
        # Iterate over each batch
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            h = batch['h'].to(device) # CLIP text encoder's last layer output text features (EOS features) [batch_size, embed_dim]
            level_h = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] EOS features for each layer of the CLIP text encoder
            l_pos = batch['l_pos'].to(device) # Affirmative text features [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # Negated object text features [batch_size, embed_dim]
            I = batch['I'].to(device) # Image features [batch_size, embed_dim]
            image_ids = batch['img_id'].to(device) # Image IDs [batch_size]
            
            unique_img_ids, remapped_ids = torch.unique(image_ids, sorted=True, return_inverse=True)
            caption_to_img = remapped_ids.cpu().numpy()
            
            # Forward pass
            if with_gt_neg is True:
                scores_T2I, scores_I2T = model(I, h, level_h, l_neg) # Use GT h_neg
            else:
                scores_T2I, scores_I2T = model(I, h, level_h) # Use lens-predicted h_neg
            
            # Map scores_T2I back to [N_caps, N_imgs] based on caption_to_img
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
            
            # Gradient clipping
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} contrastive_loss: {losses['contrastive_loss']/batch_count:.4f}")
        scheduler.step()    
        
        # Validation
        val_recall5 = evaluate_model_retrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)['mean'][5] # mean-recall@5 
        
        # Save best model
        if val_recall5 > best_recall5:
            best_recall5 = val_recall5
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with recall@5: {best_recall5}")
        else: # Early stopping
            patience_counter += 1 # Increment patience counter
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
    CCNeg Dataset:
        
    def __getitem__(self, idx):
        return {
            'Ip': self.data[idx]['I'], # Positive sample image features [embed_dim]
            'In': self.data[top1_index]['I'] # Negative sample image features [embed_dim] 
            'hp': self.data[idx]['hp'], # Affirmative text features [embed_dim]
            'hn': self.data[idx]['hn'], # Distractor text features with negation [embed_dim]
            'level_hp_list': self.data[idx]['level_hp_list'], # Affirmative text feature list (per layer) [num_layers, embed_dim]
            'level_hn_list': self.data[idx]['level_hn_list'], # Distractor text feature list with negation (per layer) [num_layers, embed_dim]
            'l_pos': self.data[idx]['l_pos'], # Affirmative text features [embed_dim]
            'l_neg': self.data[idx]['l_neg'], # Distractor text features with negation [embed_dim]
            'neg_obj': self.data[idx]['neg_obj'], # Negated object text features [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'], # Image path
            'img_id': self.data[idx]['img_id'], # Image ID
        }

    Train the Glasses model | Proxy task: CCNeg with gtneg
    
    Args:
        - cfg: Configuration file
        - model: Glasses model
        - with_gt_neg: Whether to use GT h_neg
    """
    # Read configuration
    device = cfg['device']
    epochs = cfg['epochs']
    clip_grad = True if cfg.get('clip_grad', False) else False
    batch_size = cfg['batch_size']
    early_stop_patience = cfg['early_stop_patience']
    lr = cfg['lr']
    num_workers = cfg['num_workers']

    # Create dataset and data loaders
    dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    
    train_size, val_size = len(dataset)-40000, 40000
    train_dataset = Subset(dataset, list(range(train_size))) # Training set [0, -40000)
    val_dataset = Subset(dataset, list(range(train_size, train_size+5000))) # Validation set [-40000, -1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    
    # Optimizer
    if cfg.get('only_train_moudle', None) == 'lens':
        print("Only training the lens module")
        for param in model.frame.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.lens.parameters(), lr=lr, betas=(0.9, 0.98))
    elif cfg.get('only_train_moudle', None) == 'frame':
        print("Only training the frame module")
        for param in model.lens.parameters():
            param.requires_grad = False
        optimizer = optim.AdamW(model.frame.parameters(), lr=lr, betas=(0.9, 0.98))
    else: # Train all modules
        print("Training all modules of Glasses")
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Gradient monitoring hook
    print("Registering gradient monitoring hook")
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                lambda grad, name=name: print(f"Gradient norm for {name}: {grad.norm().item():.4f}")
                if grad.norm() > 5e2 else None
            )
    
    # Pre-training evaluation
    evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
    
    # Training loop
    best_acc = 0
    patience_counter = 0
    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0
        losses = {'loss_Ip2Tpn': 0, 'loss_Tpn2Ip': 0, 'loss_In2Tnp': 0, 'loss_Tpn2In': 0}
                
        # Iterate over each batch
        for batch in tqdm(train_loader, desc=f"Epoch{epoch+1}/{epochs}"):
            Ip = batch['Ip'].to(device)  # Positive sample image features [batch_size, embed_dim]
            In = batch['In'].to(device)  # Negative sample image features [batch_size, embed_dim]
            hp = batch['hp'].to(device)  # Affirmative text features [batch_size, embed_dim]
            hn = batch['hn'].to(device)  # Distractor text features with negation [batch_size, embed_dim]
            level_hp_list = batch['level_hp_list'].to(device)  # Affirmative text feature list [batch_size, num_layers, embed_dim]
            level_hn_list = batch['level_hn_list'].to(device)  # Distractor text feature list with negation [batch_size, num_layers, embed_dim]
            neg_obj = batch['neg_obj'].to(device)  # Negated object text features [batch_size, embed_dim]
            img_id = batch['img_id'].to(device)  # Image ID [batch_size]
            
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
            
            # Each image corresponds to one hp (affirmative text) and one hn (hard negative text created by negating hp, with no matching image). 
            # Other hp and hn in the batch are regular negatives.
            loss, loss_dict = model.calc_ccneg_4_losses(scores_T2Ip, scores_Ip2T, scores_In2T, scores_T2In)
            
            epoch_loss += loss.item()

            losses['loss_Ip2Tpn'] += loss_dict['loss_Ip2Tpn']
            losses['loss_Tpn2Ip'] += loss_dict['loss_Tpn2Ip']
            losses['loss_In2Tnp'] += loss_dict['loss_In2Tnp']
            losses['loss_Tpn2In'] += loss_dict['loss_Tpn2In']
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        batch_count = len(train_loader)
        print(f"Ep{epoch+1}/{epochs}  Loss: {epoch_loss/batch_count:.4f} \
              loss_Ip2Tpn: {losses['loss_Ip2Tpn']/batch_count:.4f} \
              loss_Tpn2Ip: {losses['loss_Tpn2Ip']/batch_count:.4f} \
              loss_In2Tnp: {losses['loss_In2Tnp']/batch_count:.4f} \
              loss_Tpn2In: {losses['loss_Tpn2In']/batch_count:.4f}")
        
        scheduler.step()    
        
        # Validation
        val_results = evaluate_model_CCNeg_etrieval_withGTNeg(model, val_loader, test_raw_clip=False, with_gt_neg=with_gt_neg)
        val_acc = val_results['accuracy'] # ACC
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(current_dir, cfg['save_path']))
            print(f"Best model saved at epoch {epoch} with ACC: {best_acc}")
        else:  # Early stopping
            patience_counter += 1  # Increment patience counter
            print(f"💔ACC drop from {best_acc:.4f} to {val_acc:.4f}, cur patience_counter add to {patience_counter}")
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
                'val_acc': val_acc
            }
            torch.save(checkpoint, os.path.join(current_dir, f"checkpoint_epoch_{epoch}.pth"))
        
    print(f"Training completed. Best validation ACC: {best_acc:.4f}")
    
    return model


if __name__ == "__main__":
    cfg = {
        # -----Training Parameters-----
        'epochs': 30,
        'batch_size': 64,
        'lr': 1e-5,
        'num_workers': 4,
        'early_stop_patience': 5, # Early stopping patience
        'device': 'cuda',
        'dtype': torch.float32,
        'save_path': 'best_clip_Glasses.pth', # Path to save the trained model weights
        'pretrain': False, # Whether to use pretrained Glasses
        
        # -----Model Parameters-----
        'Lens': {
            'device': 'cuda',
            'dtype': torch.float32,
            'num_heads': 4,
            'dropout': 0.1,
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/v2_COCO/best_clip_lens_9922.pth' # Pretrained weights for Lens
        },
        'Frame': {
            'device': 'cuda',
            'dtype': torch.float32,
            'lambda_0': 1, # Base penalty strength
        },
        'NegationDetector': {
            'device': 'cuda',
            'model_path': '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth', # Pretrained weights for NegationDetector
            'neg_thr': 0.5, # Negation threshold (greater than this value indicates negation) e.g., full negation: -1.0, full affirmation: 1.0 #TODO: Parameter tuning required
        },
        
        # -----Data Parameters-----
        'Mcq': {
            'batch_size': 64,
            'num_workers': 4,
            'num_options': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv',
            'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv',
        },
        'Retrieval': {
            'batch_size': 64,
            'num_workers': 4,
            'split': [0.9, 0.1, 0.0],
            'train_dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
            'test_dataset_path': '/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv',
        },
        'RetrievalWithGtNeg': { # h_neg is directly provided as GT, only train and test the Frame
            'batch_size': 64,
            'num_workers': 4, 
            'split': [0.9, 0.1, 0.0],  # train, val, test split
            'pos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_retrieval.csv",
            'negpos_csv_path': "/root/NP-CLIP/NegBench/data/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv",
            'dtype': torch.float32, 
        },
        'CCNegGtDataset': { # CC-Neg dataset
            'batch_size': 64,
            'num_workers': 4,
            'csv_path': '/root/NP-CLIP/NegBench/data/ccneg_converted.csv',
            'negative_image_ft_mapping_path': '/root/NP-CLIP/NegBench/data/distractor_image_mapping.pt', # Hard negative sample image indices
            'dtype': torch.float32, 
            
        },
        'ClsEvalDataset': {  # Traditional classification test dataset
            'csv_path': '/root/NP-CLIP/NegBench/data/CLS_Imagenet/imagenet_train.csv',
            'batch_size': 64,
            'num_workers': 4,
        }
    }

    # # --------------------------------------------Training on COCO----------------------------------------------
    
    # Training on COCO
    # Stage 1: Use gtneg instead of lens output, train Frame model only, do not train Lens model
    cfg['lr'] = 1e-4
    cfg['neg_thr'] = -1
    cfg['epochs'] = 10
    model = Glasses.load_model(cfg)
    model = train_COCORetr_with_gtneg(cfg, model, with_gt_neg=True) # Stage 1: Train Glasses model | Proxy task: Retrieval with gtneg
    
    # Stage 2: Use GT_neg as supervision to train lens only, implemented in lens.py
    cfg['Lens']['model_path'] = 'weights/v2_COCO/best_clip_lens_9922.pth' # Path to pretrained weights for lens
    
    # Stage 3: Jointly train lens and Frame models for adaptation
    cfg['pretrain'] = True
    cfg['lr'] = 1e-3
    cfg['model_path'] = 'best_clip_Glasses.pth' # Path to Stage 1 pretrained model weights
    cfg['neg_thr'] = -1
    cfg['clip_grad'] = True # Gradient clipping
    model = Glasses.load_model(cfg)
    model.lens = CLIPGlassesLens.load_model(cfg['Lens']) # Load pretrained weights for lens
    model = train_COCORetr_with_gtneg(cfg, model, with_gt_neg=False) # Stage 2: Jointly train Glasses model with lens | Proxy task: Retrieval
    
    # # -------------------------------------------Training on CC-Neg----------------------------------------------
    
    # # Training on CC-Neg
    # # Stage 1: Use gtneg instead of lens output, train Frame model only, do not train Lens model
    # cfg['lr'] = 1e-4
    # cfg['neg_thr'] = -1 # Negation threshold
    # cfg['epochs'] = 10
    # model = Glasses.load_model(cfg)
    # model = train_CCNeg_with_gtneg(cfg, model, with_gt_neg=True) # Stage 1: Train Glasses model | Proxy task: Retrieval with gtneg
    
    # # Stage 2: Use GT_neg as supervision to train lens only, implemented in lens.py
    # cfg['Lens']['model_path'] = 'weights/v2_COCO/best_clip_lens_9922.pth' # Path to pretrained weights for lens
    
    # # Stage 3: Jointly train lens and Frame models for adaptation
    # cfg['pretrain'] = True
    # cfg['lr'] = 1e-3
    # cfg['model_path'] = 'best_clip_Glasses.pth' # Path to Stage 1 pretrained model weights
    # cfg['neg_thr'] = -1
    # cfg['clip_grad'] = True # Gradient clipping
    # model = Glasses.load_model(cfg)
    # model.lens = CLIPGlassesLens.load_model(cfg['Lens']) # Load pretrained weights for lens
    # model = train_CCNeg_with_gtneg(cfg, model, with_gt_neg=False) # Stage 2: Jointly train Glasses model with lens | Proxy task: Retrieval
    
    
    # # --------------------------------------Testing Configuration (COCO & CC-Neg)----------------------------------------------
    
    # # Testing Glasses model trained on COCO
    # cfg['test_raw_clip'] = False # Test the original CLIP model
    # cfg['test'] = True
    # cfg['model_path'] = 'weights/v2_COCO/best_clip_Glasses.pth'
    # cfg['Lens']['model_path'], cfg['Frame']['model_path'] = None, None # Do not overwrite joint-trained Glasses
    # cfg['NegationDetector']['model_path'] = '/root/NP-CLIP/XTrainer/exp/exp5_glasses/weights/best_NegDet_9404_9212.pth'
    
    # # --------------------------------------Testing Model Performance (Comparison Experiments)--------------------------------------
    
    # # Testing CC-Neg 
    # test_ccneg_dataset = CCNegGtDataset(cfg['CCNegGtDataset'])
    # train_size, val_size = len(test_ccneg_dataset)-40000, 40000
    # val_dataset = Subset(test_ccneg_dataset, list(range(train_size, train_size+val_size))) # Validation set [-40000, -1)
    # test_ccneg_dataloader = torch.utils.data.DataLoader(test_ccneg_dataset, batch_size=cfg['CCNegGtDataset']['batch_size'], shuffle=False, num_workers=cfg['CCNegGtDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CCNeg_etrieval_withGTNeg(None, val_dataset, test_raw_clip=True, with_gt_neg=False)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CCNeg_etrieval_withGTNeg(model, val_dataset, test_raw_clip=False, with_gt_neg=False) # Use lens-predicted h_neg
    
    # # Testing COCO-Retrieval with gtneg
    # test_retrieval_dataset = RetrievalNegGtDataset(cfg['RetrievalWithGtNeg'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Retrieval']['batch_size'], shuffle=False, num_workers=cfg['Retrieval']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_retrieval_withGTNeg(None, test_retrieval_dataloader, test_raw_clip=True, with_gt_neg=False)
    # else:
    #     model = Glasses.load_model(cfg)
    #     # evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False, with_gt_neg=True) # Use GT h_neg
    #     evaluate_model_retrieval_withGTNeg(model, test_retrieval_dataloader, test_raw_clip=False, with_gt_neg=False) # Use lens-predicted h_neg
        
    # # Testing MCQ VOC 
    # cfg['Mcq']['test_dataset_path'] = '/root/NP-CLIP/NegBench/data/images/MCQ/VOC2007_mcq_llama3.1_rephrased.csv'
    # test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
        
    # # Testing MCQ COCO
    # cfg['Mcq']['test_dataset_path'] = '/root/NP-CLIP/NegBench/data/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv'
    # test_retrieval_dataset = McqDataset(cfg['Mcq']['test_dataset_path'])
    # test_retrieval_dataloader = torch.utils.data.DataLoader(test_retrieval_dataset, batch_size=cfg['Mcq']['batch_size'], shuffle=False, num_workers=cfg['Mcq']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_mcq(None, test_retrieval_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_mcq(model, test_retrieval_dataloader, test_raw_clip=False)
    
    
    # # --------------------------------------Testing CLIP Traditional Zero-Shot Capability Retention--------------------------------------
    # # Testing ImageNet traditional classification capability retention
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/imagenet_val.csv'
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
        
    # # Testing Caltech101 traditional classification capability retention
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/caltech101.csv'
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
        
    # # Testing CIFAR-100 traditional classification capability retention
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/cifar100.csv'
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
        
    # # Testing CIFAR-10 traditional classification capability retention
    # cfg['ClsEvalDataset']['csv_path'] = '/root/NP-CLIP/NegBench/data/CLS/cifar10.csv'
    # test_dataset = CLSDataset(cfg['ClsEvalDataset'])
    # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg['ClsEvalDataset']['batch_size'], shuffle=False, num_workers=cfg['ClsEvalDataset']['num_workers'])
    # if cfg['test_raw_clip'] is True:
    #     evaluate_model_CLS(None, test_dataloader, test_raw_clip=True)
    # else:
    #     model = Glasses.load_model(cfg)
    #     evaluate_model_CLS(model, test_dataloader, test_raw_clip=False)
    
    
    # print("==============Configuration===============")
    # for k, v in cfg.items():
    #     if isinstance(v, dict):
    #         print(f"{k}:")
    #         for k1, v1 in v.items():
    #             print(f"  {k1}: {v1}")
    #     else:
    #         print(f"{k}: {v}")
    # print("===================================")