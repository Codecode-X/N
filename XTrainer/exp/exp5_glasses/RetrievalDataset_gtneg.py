import os
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, extract_objs_features, Clip_model


# Dataset class for training
class RetrievalNegGtDataset(Dataset):
    """ RetrievalDataset + GT negative object """
    def __init__(self, cfg):
        self.cfg = cfg
        self.pos_csv_path = cfg['pos_csv_path'] # COCO_val_retrieval.csv
        self.negpos_csv_path = cfg['negpos_csv_path'] # COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
        self.data = [] # 在_preprocess_features()中填充 | [{'h': h, 'level_h_list': level_h_list, 'l_pos': l_pos, 'img_path': img_path}), ...]
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - 如果有预处理数据文件，则直接加载
            - 如果没有则提取，并保存预处理数据文件，下次直接加载
        """
        # Create cache file path based on CSV path
        cache_path = f"RetrievalNegGtDataset_cache.pt"
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"Loading preprocessed features from cache: {cache_path}...")
            self.data = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.data)} samples from cache")
            return
        
        # Read CSV files
        df_np = pd.read_csv(self.negpos_csv_path)
        df_p = pd.read_csv(self.pos_csv_path)
        
        # Create image_id lookup dictionaries
        np_by_id = {}
        p_by_id = {}
        for _, row in df_np.iterrows():
            np_by_id[row['image_id']] = row
        for _, row in df_p.iterrows():
            p_by_id[row['image_id']] = row
        
        # Match samples and extract features
        common_ids = set(np_by_id.keys()) & set(p_by_id.keys())
        
        for img_id in tqdm.tqdm(common_ids, desc="Processing data"):
            np_row = np_by_id[img_id]
            p_row = p_by_id[img_id]
            
            np_captions = eval(np_row['captions']) # 对应 negpos_csv
            p_captions = eval(p_row['captions']) # 对应 pos_csv
            
            neg_object_list = eval(np_row['negative_objects'])
            neg_objs_list = extract_objs_features(neg_object_list) # 提取 neg_object_list 中的每个对象的文本特征 | [num_objs, embed_dim]

            for np_cap, p_cap in zip(np_captions, p_captions):
                # print(f"Processing image_id: {img_id}, np_cap: {np_cap}, p_cap: {p_cap}")
                h, level_h_list = extract_sentence_features(np_cap)
                h = torch.tensor(h, dtype=self.cfg['dtype'])
                l_pos, _ = extract_sentence_features(p_cap) # [embed_dim]
                img_path = np_row['filepath']
                I = extract_img_features(image_path=np_row['filepath'])
                I = torch.tensor(I, dtype=self.cfg['dtype']) # [embed_dim]
                
                # 求候选列表neg_objs_list中每个neg_obj与h的余弦相似度, 相似度最大的neg_obj为相应的被否定对象
                biggest_sim = -float('inf')
                correct_neg_obj, correct_neg_obj_str = None, None
                for i, neg_obj in enumerate(neg_objs_list):
                    neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype']) # [embed_dim]
                    sim = F.cosine_similarity(h, neg_obj, dim=-1)
                    if sim > biggest_sim:
                        biggest_sim = sim
                        correct_neg_obj = neg_obj
                        correct_neg_obj_str = neg_object_list[i] # 对应的 neg_object
                
                if correct_neg_obj is None: # 无否定对象
                    correct_neg_obj = torch.zeros_like(h) # 全0向量-torch.all(correct_neg_obj == 0)
                
                # print(f"img_id: {img_id}, np_cap: {np_cap}, p_cap: {p_cap}, correct_neg_obj_str: {correct_neg_obj_str}, biggest_sim: {biggest_sim}")
                self.data.append({'I': I, 'h': h, 'level_h_list': level_h_list, 'l_pos': l_pos, 'neg_obj': correct_neg_obj, 'img_path': img_path, 'img_id': img_id})
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")  
        torch.save(self.data, cache_path)
        print(f"Preprocessed features saved to {cache_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'I': self.data[idx]['I'], # 图像特征 [embed_dim]
            'h': self.data[idx]['h'], # CLIP文本编码器的输出文本特征(EOS特征)
            'level_h_list': torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in self.data[idx]['level_h_list']]),  # CLIP文本编码器每一层的EOS特征
            'l_pos': torch.tensor(self.data[idx]['l_pos'], dtype=self.cfg['dtype']),
            'neg_obj': self.data[idx]['neg_obj'].to(dtype=self.cfg['dtype']), # [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'],
            'img_id': self.data[idx]['img_id'] # 提取图像ID  
        }


def evaluate_model_retrieval_withGTNeg(model, data_loader, test_raw_clip=False, with_gt_neg=True, device='cuda'):
    """
    Evaluate the model on the retrieval task. (h_neg直接给出GT，而不是通过Lens预测)
    
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
    all_neg_feats = [] # 被否定对象的文本特征
    all_level_text_feats = [] # 原始CLIP每一层输出的文本特征列表
    caption_to_img  = []
    
    with torch.no_grad():
       # 遍历每一个batch
        for batch in tqdm(data_loader, desc=f"TESTING", total=len(data_loader)):
            caption_feats = batch['h'].to(device) # CLIP文本编码器最后一层的输出文本特征(EOS特征) [batch_size, embed_dim]
            level_H_list = batch['level_h_list'].to(device) # [batch_size, num_layers, embed_dim] CLIP文本编码器每一层的EOS特征
            l_pos = batch['l_pos'].to(device) # 肯定文本特征 [batch_size, embed_dim]
            l_neg = batch['neg_obj'].to(device) # 被否定对象的文本特征 [batch_size, embed_dim]
            image_feats = batch['I'].to(device) # 图像特征 [batch_size, embed_dim]
            image_ids = batch['img_id'].to(device) # 图像ID [batch_size]
            caption_to_img.extend(image_ids.cpu().numpy()) # [N_caps]
            all_image_feats.extend(image_feats.cpu().numpy()) # [N_imgs, embed_dim]
            all_text_feats.extend(caption_feats.cpu().numpy()) # [N_caps, embed_dim]
            all_neg_feats.extend(l_neg.cpu().numpy()) # [N_caps, embed_dim]
            all_level_text_feats.extend(level_H_list.cpu().numpy()) # [N_caps, num_layers, embed_dim]
        
        # caption_to_img 图像id转为索引 | [0, 0, 1, 1, 2, 2, ...]
        caption_to_img = torch.tensor(caption_to_img, dtype=torch.long)
        unique_img_ids, remapped_ids = torch.unique(caption_to_img, sorted=True, return_inverse=True)
        caption_to_img = remapped_ids.cpu().numpy()
        
        N_imgs = len(unique_img_ids) # 图像数量
        N_caps = len(caption_to_img)
        print(f"N_imgs: {N_imgs}, N_caps: {N_caps}")
        
        # Stack 成大 tensor
        all_image_feats = [torch.from_numpy(f) for f in all_image_feats]
        all_text_feats = [torch.from_numpy(f) for f in all_text_feats]
        all_neg_feats = [torch.from_numpy(f) for f in all_neg_feats]
        all_level_text_feats = [torch.from_numpy(f) for f in all_level_text_feats]
        I = torch.stack(all_image_feats, dim=0).to(device)  # [N_caps, D]
        I_rep = I # [N_caps, D]
        h = torch.stack(all_text_feats, dim=0).to(device)  # [N_caps, D]
        l_neg = torch.stack(all_neg_feats, dim=0).to(device) # [N_caps, D]
        level_h = torch.stack(all_level_text_feats, dim=0).to(device) # [N_caps, L, D]
      
        # ----TEST: 直接使用原始的clip输出计算----- 55.84%
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
            if with_gt_neg:
                print("使用GT neg_obj作为被否定对象的文本特征:")
                scores_T2I, scores_I2T = model(I_rep, h, level_h, l_neg) # 使用gt 57.74%
            else:
                print("使用Lens预测的neg_obj作为被否定对象的文本特征:")
                scores_T2I, scores_I2T = model(I_rep, h, level_h) # 使用lens预测h_neg 58.77%
                # zeor_neg = torch.zeros_like(l_neg)
                # print(f"使用全0向量作为被否定对象的文本特征: ", zeor_neg)
                # scores_T2I, scores_I2T = model(I_rep, h, level_h, zeor_neg) # 使用zero # 55.86%
        
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