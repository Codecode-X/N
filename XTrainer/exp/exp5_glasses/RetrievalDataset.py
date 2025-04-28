import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, Clip_model


class RetrievalDataset(Dataset):
    """
    Dataset for retrieval task
    """
    def __init__(self, csv_path, transform=None, device='cuda'):
        """
        Args:
            csv_path: Path to the CSV file with MCQ data
            clip_model: CLIP model for feature extraction
            lens_model: CLIPGlassesLens model for feature extraction
            transform: Optional transform to be applied on images
        """
        self.device = device
        self.csv_path = csv_path
        try:
            self.data = pd.read_csv(csv_path, encoding='gbk')
        except UnicodeDecodeError:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        self.transform = transform
        
        # Preprocess all data
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - Load from cache if available
            - Otherwise extract features and save to cache
        """
        # Create cache file path based on CSV path
        csv_hash = hashlib.md5(self.data.to_json().encode()).hexdigest()[:10]
        cache_path = f"retrieve_features_cache_{csv_hash}.pt"
        
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.image_features = cached_data['image_features']
            self.captions_feats = cached_data['captions_feats']
            self.level_H = cached_data['level_H_list']
            self.image_ids = cached_data['image_ids']
            print(f"Loaded {len(self.image_features)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        
        self.image_features = []
        self.captions_feats = [] # 原始CLIP输出的文本特征
        self.level_H = [] # 原始CLIP每一层输出的图像特征列表
        self.image_ids = []
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            img_path = row['filepath']
            image_id = row['image_id']
            captions = eval(row['captions']) # 每个图片对应的数量不一致

            # 提取图像特征
            img_feature = extract_img_features(img_path) # 原始CLIP提取的图像特征
            self.image_features.append(img_feature)
            self.image_ids.append(image_id)

            row_h = []
            row_level_h = []
            for caption in captions:
                h, level_h_list = extract_sentence_features(caption) # 原始CLIP提取的文本特征 | h:[embed_dim], level_h_list:[embed_dim]*num_layers
                text_tensor = torch.tensor(h, dtype=torch.float32).to(self.device)
                row_h.append(text_tensor.cpu().numpy()) 
                row_level_h.append(level_h_list)
            self.captions_feats.append(row_h)
            self.level_H.append(row_level_h)

        print(f"Saving features to cache: {cache_path}")
        torch.save({
            'image_features': self.image_features,
            'captions_feats': self.captions_feats,
            'level_H_list': self.level_H,
            'image_ids': self.image_ids
        }, cache_path)
        print(f"Cached {len(self.image_ids)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            image_features: Image features [embed_dim]
            caption_feat: Text features [num_captions_i, embed_dim]
            level_h_list: List of text features [num_captions_i, num_layers, embed_dim]
            image_id: Image ID
        """
        img_feature = torch.tensor(self.image_features[idx], dtype=torch.float32)
        caption_feat = torch.from_numpy(np.array(self.captions_feats[idx])).float() # [num_captions_i, embed_dim]
        level_h_list = torch.from_numpy(np.array(self.level_H[idx])).float() # [num_captions_i, num_layers, embed_dim]
        image_id = torch.tensor(int(self.image_ids[idx]))
        
        return img_feature, caption_feat, level_h_list, image_id


def retrieval_collate_fn(batch):
    img_features, pos_features, neg_features, image_ids = zip(*batch)
    return (
        torch.stack(img_features),  # [B, embed_dim]
        list(pos_features),         # list of [num_captions_i, embed_dim]
        list(neg_features),         # list of [num_captions_i, embed_dim]
        torch.stack(image_ids)      # [B]
    )  
        
def evaluate_model_retrieval(model, data_loader, test_raw_clip=False, device='cuda'):
    """
    Evaluate the model on the retrieval task.
    
    参数：
        - model: 待测试模型
        - data_loader: 验证数据加载器
        - device: 设备（CPU或GPU）
    
    返回：
        - results: 评估结果字典，包含文本到图像和图像到文本的召回率
            - txt2img: 文本到图像的召回率 | txt2img[k]表示recall@k
            - img2txt: 图像到文本的召回率 | img2txt[k]表示recall@k
            - mean: 平均召回率 | mean[k]表示recall@k
    """
    # Metrics
    txt2img_recalls = {1: 0, 5: 0, 10: 0}
    img2txt_recalls = {1: 0, 5: 0, 10: 0}
    all_image_feats = []
    all_text_feats  = [] # 原始CLIP最终输出的文本特征
    all_level_text_feats = [] # 原始CLIP每一层输出的文本特征列表
    caption_to_img  = []
    
    with torch.no_grad():
        for image_feats, caption_feats, level_H_list, image_ids in tqdm(data_loader, desc="Extract feats"):
            """
                image_feats: [B, D]
                caption_feats: [B, num_caps_i, D]
                level_H_list:  [B, num_caps_i, L, D]
            """
            B = image_feats.size(0)
            for batch_idx in range(B):
                # 把这张图 append 到 all_image_feats
                all_image_feats.append(image_feats[batch_idx].cpu())
                img_idx = len(all_image_feats)-1 # 这张图在 list 中的新索引
                # 把这张图的每条 caption 都收集起来，并记录它们对应 img_idx
                caps_b = caption_feats[batch_idx] # [num_caps_i, D]
                levels_b = level_H_list[batch_idx] # [num_caps_i, L, D]
                for cap_f, lvl_f in zip(caps_b, levels_b):
                    all_text_feats.append(cap_f.cpu())  # [D]
                    all_level_text_feats.append(lvl_f.cpu()) # [L, D]
                    caption_to_img.append(img_idx) # 记录 caption 属于哪张图

        # Stack 成大 tensor
        I = torch.stack(all_image_feats, dim=0).to(device)  # [N_imgs, D]
        h = torch.stack(all_text_feats, dim=0).to(device)  # [N_caps, D]
        level_h = torch.stack(all_level_text_feats, dim=0).to(device) # [N_caps, L, D]
        N_imgs, N_caps = I.size(0), h.size(0)
        
        # 构造一对一的 I_rep，使其和 h/level_h 在 batch 维度上对齐
        idx = torch.tensor(caption_to_img, dtype=torch.long, device=device) # [N_caps]
        I_rep = I[idx]  # [N_caps, D]
        
        # ----TEST: 直接使用原始的clip输出计算-----
        if test_raw_clip:
            print("直接使用原始的clip输出计算:")
            I_norm = F.normalize(I_rep, p=2, dim=-1)
            h_norm = F.normalize(h, p=2, dim=-1)
            logit_scale = Clip_model.logit_scale.exp()
            scores_T2I = logit_scale * (h_norm @ I_norm.t())
            scores_I2T = scores_T2I.t()
        # ---------------------------------------
        else: # 使用当前模型
            model.eval()
            scores_T2I, scores_I2T = model(I_rep, h, level_h)
        
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
                
        # 评估 Recall@1/5/10  
        txt2img_hits = {1:0, 5:0, 10:0}
        img2txt_hits = {1:0, 5:0, 10:0}

        # Text→Image
        for cap_idx, gt_img in enumerate(caption_to_img):
            scores = scores_T2I[cap_idx] # [N_imgs=149]
            # top-k 图像索引
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in txt2img_hits:
                if gt_img in topk[:k]:
                    txt2img_hits[k] += 1

        # Image→Text
        # 先构造每张图的 caption 索引列表
        img2cap = [[] for _ in range(N_imgs)]  # 例如访问 img2cap[0]，得到图像 0 的所有 caption 索引
        for cap_idx, img_idx in enumerate(caption_to_img):
            img2cap[img_idx].append(cap_idx)

        for img_idx in range(N_imgs):
            if not img2cap[img_idx]:
                continue
            scores = scores_I2T[img_idx] # [N_caps]
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in img2txt_hits:
                # 只要有任一 gt caption 在 top-k 中，就算命中
                if any(cap in topk[:k] for cap in img2cap[img_idx]):
                    img2txt_hits[k] += 1

    # 计算并打印百分比 
    total_caps = float(N_caps)
    total_imgs = float(N_imgs)
    txt2img_recalls = {k: txt2img_hits[k]/total_caps*100 for k in txt2img_hits}
    img2txt_recalls = {k: img2txt_hits[k]/total_imgs*100 for k in img2txt_hits}

    print("\nText→Image Retrieval:")
    for k,v in txt2img_recalls.items():
        print(f"  Recall@{k}: {v:.2f}%")
    print("\nImage→Text Retrieval:")
    for k,v in img2txt_recalls.items():
        print(f"  Recall@{k}: {v:.2f}%")
    print("\nMean Retrieval:")
    for k in [1,5,10]:
        print(f"  Recall@{k}: {(txt2img_recalls[k]+img2txt_recalls[k])/2:.2f}%")

    return {
        'txt2img': txt2img_recalls,
        'img2txt': img2txt_recalls,
        'mean':   {k:(txt2img_recalls[k]+img2txt_recalls[k])/2 for k in txt2img_recalls}
    }