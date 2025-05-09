from get_model import extract_sentence_features, extract_objs_features, extract_img_features
import torch
import tqdm
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
import os

# Dataset class for training
class GlassesDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.pos_csv_path = cfg['pos_csv_path'] # COCO_val_retrieval.csv
        self.negpos_csv_path = cfg['negpos_csv_path'] # COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
        self.data = [] # Filled in _preprocess_features() | [{'h': h, 'level_h_list': level_h_list, 'l_pos': l_pos, 'img_path': img_path}), ...]
        self._preprocess_features()
        
    def _preprocess_features(self):
        """
        Preprocess and cache all image and text features
            - If a preprocessed data file exists, load it directly
            - If not, extract features, save the preprocessed data file, and load it next time
        """
        # Create cache file path based on CSV path
        cache_path = f"LensDataset_cache.pt"
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
            
            np_captions = eval(np_row['captions']) # Corresponding to negpos_csv
            p_captions = eval(p_row['captions']) # Corresponding to pos_csv
            
            neg_object_list = eval(np_row['negative_objects'])
            neg_objs_list = extract_objs_features(neg_object_list) # Extract text features for each object in neg_object_list | [num_objs, embed_dim]

            for np_cap, p_cap in zip(np_captions, p_captions):
                # print(f"Processing image_id: {img_id}, np_cap: {np_cap}, p_cap: {p_cap}")
                h, level_h_list = extract_sentence_features(np_cap)
                h = torch.tensor(h, dtype=self.cfg['dtype'])
                l_pos, _ = extract_sentence_features(p_cap) # [embed_dim]
                img_path = np_row['filepath']
                I = extract_img_features(image_path=np_row['filepath'])
                I = torch.tensor(I, dtype=self.cfg['dtype']) # [embed_dim]
                
                # Compute cosine similarity between h and each neg_obj in the candidate list neg_objs_list. The neg_obj with the highest similarity is the corresponding negated object.
                biggest_sim = -float('inf')
                correct_neg_obj, correct_neg_obj_str = None, None
                for i, neg_obj in enumerate(neg_objs_list):
                    neg_obj = torch.tensor(neg_obj, dtype=self.cfg['dtype']) # [embed_dim]
                    sim = F.cosine_similarity(h, neg_obj, dim=-1)
                    if sim > biggest_sim:
                        biggest_sim = sim
                        correct_neg_obj = neg_obj
                        correct_neg_obj_str = neg_object_list[i] # Corresponding neg_object
                
                if correct_neg_obj is None: # No negated object
                    correct_neg_obj = torch.zeros_like(h) # Zero vector - torch.all(correct_neg_obj == 0)
                
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
            'I': self.data[idx]['I'], # Image features [embed_dim]
            'h': self.data[idx]['h'], # Text features (EOS features) from the CLIP text encoder
            'level_h_list': torch.stack([torch.tensor(l, dtype=self.cfg['dtype']) for l in self.data[idx]['level_h_list']]),  # EOS features from each layer of the CLIP text encoder
            'l_pos': torch.tensor(self.data[idx]['l_pos'], dtype=self.cfg['dtype']),
            'neg_obj': self.data[idx]['neg_obj'].to(dtype=self.cfg['dtype']), # [num_objs, embed_dim]
            'img_path': self.data[idx]['img_path'],
            'img_id': self.data[idx]['img_id'] # Extracted image ID  
        }
