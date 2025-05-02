import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from get_model import extract_img_features, extract_sentence_features, extract_objs_features, Clip_model
import numpy as np

# Dataset class for training
class CCNegGtDataset(Dataset):
    """ CCNegDataset + GT negative object + negative photo """
    def __init__(self, cfg):
        self.cfg = cfg
        self.csv_path = cfg['csv_path'] # COCO_val_retrieval.csv
        self.data = [] # 在_preprocess_features()中填充
        
        self.negatives_mapping_cc3m_to_coco = torch.load(cfg['negative_image_ft_mapping_path'], weights_only=False) # 否定文本对应的负样本图像索引
        
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - 如果有预处理数据文件，则直接加载
            - 如果没有则提取，并保存预处理数据文件，下次直接加载
        """
        # Create cache file path based on CSV path
        cache_path = f"CCNegGtDataset_cache.pt"
        # Check if cache file exists
        if os.path.exists(cache_path):
            print(f"正在加载CC-Neg数据集 cache: {cache_path}...")
            self.data = torch.load(cache_path, weights_only=False)
            print(f"Loaded {len(self.data)} samples from cache")
            return
        
        # Read CSV files
        df = pd.read_csv(self.csv_path)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing features"):
            # str
            caption_p = eval(row['captions'])[0] # 肯定文本
            caption_n = eval(row['n_captions'])[0] # 否定文本
            img_path = row['filepath']
            image_id = row['image_id']
            neg_object = eval(row['negative_objects'])[0]
            # 提取特征
            I = extract_img_features(image_path=img_path) # [embed_dim]
            hp, level_hp_list = extract_sentence_features(caption_p) # [embed_dim] # 肯定文本特征
            hn, level_hn_list = extract_sentence_features(caption_n) # [embed_dim] # 否定文本特征
            neg_obj = extract_objs_features([neg_object])[0] # [num_objs, embed_dim]
            # 转为 tensor
            I = torch.tensor(I, dtype=self.cfg['dtype']) # [embed_dim]
            hp = torch.tensor(hp, dtype=self.cfg['dtype']) # [embed_dim]
            hn = torch.tensor(hn, dtype=self.cfg['dtype']) # [embed_dim]
            level_hp_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hp_list]) # [num_layers, embed_dim]
            level_hn_list = torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in level_hn_list]) # [num_layers, embed_dim]
            neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype']) # [embed_dim]
            
            # 维度验证
            assert I.shape == (512,), f"图像特征维度异常: {I.shape}"
            assert hp.shape == (512,), f"正文本特征维度异常: {hp.shape}"
            assert hn.shape == (512,), f"负文本特征维度异常: {hn.shape}"

            # Append to data list
            self.data.append({'I': I, 'hp': hp, 'hn': hn, 'level_hp_list': level_hp_list, 'level_hn_list': level_hn_list, 'l_pos': hp, 'l_neg': hn, 'neg_obj': neg_obj, 'img_path': img_path, 'img_id': image_id})
        
        # Save preprocessed features to cache
        print(f"Saving preprocessed features to cache: {cache_path}")  
        torch.save(self.data, cache_path)
        print(f"Preprocessed features saved to {cache_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 根据negatives_mapping_cc3m_to_coco得到负文本hn对应的负样本图像索引
        topk_indices = self.negatives_mapping_cc3m_to_coco[idx]
        top1_index = topk_indices[0]
        
        return {
            'Ip': self.data[idx]['I'], # 图像特征 [embed_dim]
            'In': self.data[top1_index]['I'], # 负样本图像特征 [embed_dim]   
            'hp': self.data[idx]['hp'], # 肯定文本特征 [embed_dim]
            'hn': self.data[idx]['hn'], # 加了否定词的干扰错误文本特征 [embed_dim]
            'level_hp_list': self.data[idx]['level_hp_list'], # (每层)否定文本特征列表 [num_layers, embed_dim]
            'level_hn_list': self.data[idx]['level_hn_list'], # (每层)加了否定词的干扰错误文本特征列表 [num_layers, embed_dim]
            'l_pos': self.data[idx]['l_pos'], # 肯定文本特征 [embed_dim]
            'l_neg': self.data[idx]['l_neg'], # 加了否定词的干扰错误文本特征 [embed_dim]
            'neg_obj': self.data[idx]['neg_obj'], # 否定对象的文本特征 [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'], # 图像路径
            'img_id': self.data[idx]['img_id'] # 图像ID
        }
        
        
def evaluate_model_CCNeg_etrieval_withGTNeg(model, data_loader, test_raw_clip=False, with_gt_neg=True, device='cuda'):
    """
    评估model在CCNeg数据集上的检索任务性能指标：
        - 图片检索文本的 Accuracy，每个图片对应一个hp(正确匹配文本)和hn(加了否定词的干扰错误文本)，此处Accuracy为在hp和hn中二选一的准确率
        - 每个图片的 Sp(每个图片与hp的相似度) 与 Sn(每个图片与hn的相似度) 之间的差值的均值
    """
    # 收集所有特征
    all_image_feats = []
    all_hp_feats = []  # 肯定文本特征
    all_hn_feats = []  # 否定文本特征
    all_level_hp_feats = []  # 每层肯定文本特征
    all_level_hn_feats = []  # 每层否定文本特征
    all_neg_obj_feats = []  # 否定对象特征
    
    with torch.no_grad():
        # 遍历每一个batch收集特征
        for batch in tqdm(data_loader, desc="Extracting features", total=len(data_loader)):
            image_feats = batch['Ip'].to(device)  # [batch_size, embed_dim]
            hp_feats = batch['hp'].to(device)    # [batch_size, embed_dim]
            hn_feats = batch['hn'].to(device)    # [batch_size, embed_dim]
            level_hp_list = batch['level_hp_list'].to(device)  # [batch_size, num_layers, embed_dim]
            level_hn_list = batch['level_hn_list'].to(device)  # [batch_size, num_layers, embed_dim]
            neg_obj_feats = batch['neg_obj'].to(device)  # [batch_size, embed_dim]
            img_ids = batch['img_id']
            
            all_image_feats.append(image_feats) # [batch_size, embed_dim] * batch_num
            all_hp_feats.append(hp_feats)
            all_hn_feats.append(hn_feats)
            all_level_hp_feats.append(level_hp_list) # [batch_size, num_layers, embed_dim] * batch_num
            all_level_hn_feats.append(level_hn_list)
            all_neg_obj_feats.append(neg_obj_feats)
    
        I = torch.cat(all_image_feats, dim=0) # [N, embed_dim]
        hp = torch.cat(all_hp_feats, dim=0) # [N, embed_dim]
        hn = torch.cat(all_hn_feats, dim=0) # [N, embed_dim]
        assert not torch.equal(hp, hn), print(f"hp and hn should not be equal, \nhp = {hp}, \nhn = {hn}")
        level_hp = torch.cat(all_level_hp_feats, dim=0) # [N, num_layers, embed_dim]
        level_hn = torch.cat(all_level_hn_feats, dim=0) # [N, num_layers, embed_dim]
        neg_obj = torch.cat(all_neg_obj_feats, dim=0) # [N, embed_dim]
        
        # 计算相似度分数
        if test_raw_clip:
            # 使用原始CLIP计算相似度
            print("测试原始CLIP模型")
            I_norm = F.normalize(I, p=2, dim=-1)
            hp_norm = F.normalize(hp, p=2, dim=-1)
            hn_norm = F.normalize(hn, p=2, dim=-1)
            
            logit_scale = Clip_model.logit_scale.exp()
            scores_I_hp = logit_scale * (I_norm @ hp_norm.t()) # I2T [num_images=N, num_texts=N]
            scores_I_hn = logit_scale * (I_norm @ hn_norm.t())  
        else:
            # 使用提供的模型计算相似度
            print("测试Glasses模型")
            model.eval()
            if with_gt_neg:
                # 使用GT否定对象
                _, scores_I_hp = model(I, hp, level_hp, neg_obj, chunk_size=-1) # I2T [num_images=N, num_texts=N]
                _, scores_I_hn = model(I, hn, level_hn, neg_obj, chunk_size=-1) 
            else:
                # 使用模型预测的否定对象
                _, scores_I_hp = model(I, hp, level_hp, chunk_size=-1) # I2T [num_images=N, num_texts=N]
                _, scores_I_hn = model(I, hn, level_hn, chunk_size=-1) 
        
        # 计算每个图像在全集所有正负文本中的recall@1，recall@5，recall@10
        num_images = I.size(0)
        
        assert scores_I_hp.shape == (num_images, num_images), f"维度异常: {scores_I_hp.shape}"
        assert scores_I_hn.shape == (num_images, num_images), f"维度异常: {scores_I_hn.shape}"

        # 合并所有文本特征和分数
        scores_combined = torch.cat([scores_I_hp, scores_I_hn], dim=1) # I2T [num_images=N, num_texts=2N]
        scores_combined = scores_combined.to(device)
        
        # 对于每个图像，其正确匹配的文本索引应该是i
        correct_indices = torch.arange(num_images, device=device)
        
        # 计算准确率：图像i的最高分数应该是与正文本hp[i]
        _, top_indices = scores_combined.topk(k=1, dim=1)
        global_r1 = (top_indices.squeeze() == correct_indices).float().mean().item() * 100
        
        # 计算每个图像在全集所有正负文本中的recall@5，recall@10
        _, top_indices = scores_combined.topk(k=5, dim=1)
        global_r5 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
        
        _, top_indices = scores_combined.topk(k=10, dim=1)
        global_r10 = (top_indices == correct_indices.unsqueeze(1)).float().sum(dim=1).mean().item() * 100
          
        # 计算每个图像的Sp与Sn之差的均值和预测准确率
        diag_scores_hp = torch.diag(scores_I_hp)  # [N]
        diag_scores_hn = torch.diag(scores_I_hn)  # [N]
        mean_diff = (diag_scores_hp - diag_scores_hn).mean().item()
        accuracy = (diag_scores_hp > diag_scores_hn).float().mean().item() * 100
        
        # 输出结果
        print("="*50)
        print(f"global Retrieval Recall@1,5,10: {global_r1:.2f}% {global_r5:.2f}% {global_r10:.2f}%")
        print(f"Select Accuracy: {accuracy:.2f}%")
        print(f"Mean Similarity Difference (Sp-Sn): {mean_diff:.4f}")
        
        return {
            'global_R1': global_r1,
            'global_R5': global_r5,
            'global_R10': global_r10,
            'accuracy': accuracy,
            'mean_diff': mean_diff
        }