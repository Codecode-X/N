"""
#####  实验2：注意力权重可视化
​	 观察CLIP文本编码器是否关注否定词（如“没有”）。
   * 提取每一层的注意力权重，并可视化**注意力热力图**。
   * 若否定词（“没有”）对主语（“猫”）有高注意力权重：说明模型捕捉到否定逻辑。
   * 若注意力分散或无聚焦：说明否定信息未被显式编码。
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from utils import load_yaml_config
from model import build_model
from analyze_tools.visualization import display_tokens_contribution


"""可视化文本注意力"""
cfg_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"
texts = [
   "A sofa with no cat",
   "A sofa with a cat",
   "A black cat on the couch",
   "A white cat on the couch"]

for text in texts:
   output_path = "output"  # 输出路径
   cfg = load_yaml_config(cfg_path) # 读取配置
   # 加载模型
   model = build_model(cfg)  
   # 可视化输入文本每个token对文本特征的贡献
   display_tokens_contribution(model, text, visualize=True, output_path=output_path)  # 可视化文本注意力