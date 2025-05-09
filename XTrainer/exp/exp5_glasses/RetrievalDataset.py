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
            print(f"Loading Retrieval dataset cache: {cache_path} of {self.csv_path} ...")
            cached_data = torch.load(cache_path, weights_only=False)
            self.image_features = cached_data['image_features']
            self.captions_feats = cached_data['captions_feats']
            self.level_H = cached_data['level_H_list']
            self.image_ids = cached_data['image_ids']
            print(f"Loaded {len(self.image_features)} samples from cache")
            return
        
        print(f"Preprocessing dataset features of {self.csv_path} ...")
        
        self.image_features = []
        self.captions_feats = [] # Original CLIP text features
        self.level_H = [] # List of image features from each layer of original CLIP
        self.image_ids = []
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            img_path = row['filepath']
            image_id = row['image_id']
            captions = eval(row['captions']) # Number of captions per image may vary

            # Extract image features
            img_feature = extract_img_features(img_path) # Image features extracted by original CLIP
            self.image_features.append(img_feature)
            self.image_ids.append(image_id)

            row_h = []
            row_level_h = []
            for caption in captions:
                h, level_h_list = extract_sentence_features(caption) # Text features extracted by original CLIP | h:[embed_dim], level_h_list:[embed_dim]*num_layers
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
    
    Args:
        - model: Model to be evaluated
        - data_loader: Validation data loader
        - device: Device (CPU or GPU)
    
    Returns:
        - results: Dictionary of evaluation results, including recall rates for text-to-image and image-to-text retrieval
            - txt2img: Text-to-image recall rates | txt2img[k] represents recall@k
            - img2txt: Image-to-text recall rates | img2txt[k] represents recall@k
            - mean: Mean recall rates | mean[k] represents recall@k
    """
    # Metrics
    txt2img_recalls = {1: 0, 5: 0, 10: 0}
    img2txt_recalls = {1: 0, 5: 0, 10: 0}
    all_image_feats = []
    all_text_feats  = [] # Final text features from original CLIP
    all_level_text_feats = [] # List of text features from each layer of original CLIP
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
                # Append this image to all_image_feats
                all_image_feats.append(image_feats[batch_idx].cpu())
                img_idx = len(all_image_feats)-1 # New index of this image in the list
                # Collect all captions for this image and record their corresponding img_idx
                caps_b = caption_feats[batch_idx] # [num_caps_i, D]
                levels_b = level_H_list[batch_idx] # [num_caps_i, L, D]
                for cap_f, lvl_f in zip(caps_b, levels_b):
                    all_text_feats.append(cap_f.cpu())  # [D]
                    all_level_text_feats.append(lvl_f.cpu()) # [L, D]
                    caption_to_img.append(img_idx) # Record which image the caption belongs to

        # Stack into large tensors
        I = torch.stack(all_image_feats, dim=0).to(device)  # [N_imgs, D]
        h = torch.stack(all_text_feats, dim=0).to(device)  # [N_caps, D]
        level_h = torch.stack(all_level_text_feats, dim=0).to(device) # [N_caps, L, D]
        N_imgs, N_caps = I.size(0), h.size(0)
        
        # Construct one-to-one I_rep to align with h/level_h in the batch dimension
        idx = torch.tensor(caption_to_img, dtype=torch.long, device=device) # [N_caps]
        I_rep = I[idx]  # [N_caps, D]
        
        # ----TEST: Directly use raw CLIP output-----
        if test_raw_clip:
            print("Directly using raw CLIP output:")
            I_norm = F.normalize(I_rep, p=2, dim=-1)
            h_norm = F.normalize(h, p=2, dim=-1)
            logit_scale = Clip_model.logit_scale.exp()
            scores_T2I = logit_scale * (h_norm @ I_norm.t())
            scores_I2T = scores_T2I.t()
        # -------------------------------------------
        else: # Use the current model
            model.eval()
            scores_T2I, scores_I2T = model(I_rep, h, level_h)
        
        # Restore scores_T2I from [N_caps, N_imgs] to [N_caps, N_imgs] based on caption_to_img
        cti = torch.tensor(caption_to_img, dtype=torch.long, device=device)  # [N_caps]
        unique_vals = torch.unique(cti, sorted=True)
        first_idx = []
        for val in unique_vals:
            idx = (cti == val).nonzero(as_tuple=True)[0][0]
            first_idx.append(idx)
        first_idx = torch.stack(first_idx, dim=0)  # [N_imgs]
        
        scores_T2I = scores_T2I[:, first_idx]  # [N_caps, N_imgs]
        scores_I2T = scores_T2I.t()
                
        # Evaluate Recall@1/5/10  
        txt2img_hits = {1:0, 5:0, 10:0}
        img2txt_hits = {1:0, 5:0, 10:0}

        # Text→Image
        for cap_idx, gt_img in enumerate(caption_to_img):
            scores = scores_T2I[cap_idx] # [N_imgs=149]
            # Top-k image indices
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in txt2img_hits:
                if gt_img in topk[:k]:
                    txt2img_hits[k] += 1

        # Image→Text
        # First construct a list of caption indices for each image
        img2cap = [[] for _ in range(N_imgs)]  # For example, accessing img2cap[0] gives all caption indices for image 0
        for cap_idx, img_idx in enumerate(caption_to_img):
            img2cap[img_idx].append(cap_idx)

        for img_idx in range(N_imgs):
            if not img2cap[img_idx]:
                continue
            scores = scores_I2T[img_idx] # [N_caps]
            neg_text_feats, topk = torch.topk(scores, k=10, largest=True)
            for k in img2txt_hits:
                # If any ground truth caption is in the top-k, count it as a hit
                if any(cap in topk[:k] for cap in img2cap[img_idx]):
                    img2txt_hits[k] += 1

    # Calculate and print percentages 
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