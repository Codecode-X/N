import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import build_model
from utils import load_yaml_config, standard_image_transform
import torch
from model.Clip import tokenize  # 分词器
import numpy as np
from torchvision.transforms import Compose, ToTensor
from PIL import Image

def build_clip_model(config_path):
    """
    构建CLIP模型
    """
    cfg = load_yaml_config(config_path, save=None, modify_fn=None) 
    model = build_model(cfg)
    return model

config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB32-ep10-Caltech101-AdamW.yaml"

Clip_model = build_clip_model(config_path=config_path) # 加载CLIP模型

def extract_sentence_features(sentence:str):
    """
    提取单个句子的CLIP文本特征
    
    参数：
        - sentence: 输入句子(str)
        
    返回：
        - text_features(torch.Tensor): CLIP文本编码器的输出文本特征(EOS特征) [embed_dim]
        - level_text_features_list(list<torch.Tensor>): CLIP文本编码器每一层的EOS特征 [num_layers, embed_dim]
    """
    with torch.no_grad():  # 关闭梯度计算
        tokenized_text = tokenize(sentence) # [num_classes=1, context_length]
        tokenized_text = tokenized_text.to(Clip_model.device) # [num_classes=1, context_length]
        text_features, level_text_features_list = Clip_model.encode_text(tokenized_text) # [num_classes=1, embed_dim]*num_layers
        text_features = text_features.cpu().numpy()[0]
        level_text_features_list = [level_text_features.cpu().numpy()[0] for level_text_features in level_text_features_list]
        return text_features, level_text_features_list # [embed_dim], [embed_dim]*num_layers


def extract_objs_features(objs:list):
    """
    提取单个句子的CLIP文本特征
    
    参数：
        - objs: 对象列表(list<str>)
        
    返回：
        # - objs_features: CLIP文本编码器的输出文本特征(EOS特征) [num_objs, embed_dim]
        - objs_features(list<torch.Tensor>: CLIP文本编码器的输出文本特征(EOS特征) [embed_dim]*num_objs
    """
    objs_features = []
    for obj in objs:
        text = f"This image shows a {obj}."
        feature, _ = extract_sentence_features(text)
        # feature = torch.tensor(feature, dtype=Clip_model.dtype) # [embed_dim]
        objs_features.append(feature)
    # objs_features = torch.stack(objs_features) # [num_objs, embed_dim]
    return objs_features


def extract_img_features(image_path:str):
    """原始CLIP提取单个图像的CLIP图像特征"""
    # 加载图像，转为张量[batch_size, 3, input_size, input_size]
    img = Image.open(image_path)  # 打开图像
    transform = Compose([standard_image_transform(224, 'BICUBIC'), ToTensor()])  # 定义转换
    img = transform(img)  # 转换图像
    img = img.unsqueeze(0)  # 添加 batch 维度
    img = img.to(device=Clip_model.device)  # 转移到模型的设备上
    with torch.no_grad():  # 关闭梯度计算
        image_features = Clip_model.encode_image(img) # [num_classes, embed_dim]
        return image_features.cpu().numpy()[0] # [embed_dim]

