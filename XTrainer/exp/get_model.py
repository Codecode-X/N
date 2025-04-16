import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import build_model
from utils import load_yaml_config

def build_clip_model(config_path):
    """
    构建CLIP模型
    """
    cfg = load_yaml_config(config_path, save=None, modify_fn=None) 
    model = build_model(cfg)
    return model