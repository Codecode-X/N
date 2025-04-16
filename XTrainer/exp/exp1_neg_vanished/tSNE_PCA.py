"""
##### 数据集

- **构造三类句子对**（每类200对）：
  1. **否定-肯定对**：
     - 否定句：`"A sofa with no cat"`
     - 肯定句：`"A sofa with a cat"`
     - **特点**：语义相似，仅否定词差异。
  2. **语义相似对**：
     - 句子1：`"A black cat on the couch"`
     - 句子2：`"A white cat on the couch"`
     - **特点**：替换非关键形容词，保持主体语义一致。
  3. **语义无关对**：
     - 句子1：`"A dog running in the park"`
     - 句子2：`"A cloud floating in blue sky"`
     - **特点**：语义完全无关。
     
##### 实验3：特征空间可视化

​	使用t-SNE将高维特征 `h` 降维至2D，绘制分布图。

* 若否定句（neg）和肯定句（pos）在特征空间中混叠：说明否定信息丢失。
* 若否定句（neg）与语义无关句（unrelated）在特征空间中混叠：说明CLIP完全无法区分否定逻辑。

"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import build_clip_model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from model.Clip import tokenize

config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"
output_dir = "/root/NP-CLIP/XTrainer/output"

Clip_model = build_clip_model(config_path=config_path)


def visualize_features(features, labels, method='tsne', title="Feature Visualization"):
    """
    对 CLIP 提取的句子特征进行降维并可视化
    :param features: numpy array [num_samples, embed_dim]
    :param labels: 0/1 标签数组
    :param method: "tsne" 或 "pca"
    :param title: 图标题
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    print(f"正在使用 {method.upper()} 降维...")
    reduced = reducer.fit_transform(features)

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.scatter(
        reduced[labels == 0, 0], reduced[labels == 0, 1],
        c='blue', label='Positive (No Negation)', alpha=0.6
    )
    plt.scatter(
        reduced[labels == 1, 0], reduced[labels == 1, 1],
        c='red', label='Negative (With Negation)', alpha=0.6
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def read_negpos_dataset(path):
    """
    读取数据集
    :param path: 数据集路径
    :return: neg_pos_pairs

    数据格式示例:
    {
        "negation_pairs": [
            {
            "negative": "A bookshelf without any novels",
            "positive": "A bookshelf with several novels",
            "type": "negation_pair",
            "key_diff": ["without any <-> with several"]
            },
            {
            "negative": "The fridge is devoid of vegetables",
            "positive": "The fridge contains fresh vegetables",
            "type": "negation_pair",
            "key_diff": ["devoid of <-> contains"]
            },...]
    }
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    neg_pos_pairs = [
        (item["negative"], item["positive"]) for item in data.get("negation_pairs", [])
    ]
    print(f"读取到 {len(neg_pos_pairs)} 对否定-肯定句子对")
    return neg_pos_pairs

# 读取语义相似对
def read_similar_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = [(item["sentence1"], item["sentence2"]) for item in data.get("semantic_similar_pairs", [])]
    print(f"读取到 {len(pairs)} 对语义相似句子对")
    return pairs

# 读取语义无关对
def read_unrelated_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = [(item["sentence1"], item["sentence2"]) for item in data.get("semantic_unrelated_pairs", [])]
    print(f"读取到 {len(pairs)} 对语义无关句子对")
    return pairs

def extract_clip_features(sentences):
    """提取句子的CLIP文本特征"""
    features = []
    with torch.no_grad():  # 关闭梯度计算
        for sentence in tqdm(sentences):
            tokenized_texts = tokenize(sentence) # [num_classes, context_length]
            tokenized_texts = tokenized_texts.to(Clip_model.device) # [num_classes, context_length]
            text_features = Clip_model.encode_text(tokenized_texts) # [num_classes, embed_dim]
            features.append(text_features.cpu().numpy()[0])

    return np.array(features)

# 构造可视化实验的主函数
def run_visualization_experiment(methods=['pca', 'tsne']):
    neg_path = "/root/NP-CLIP/XTrainer/exp/exp1_neg_vanished/neg_pos_pairs_166.json"
    sim_path = "/root/NP-CLIP/XTrainer/exp/exp1_neg_vanished/semantic_similar_pairs_130.json"
    unrel_path = "/root/NP-CLIP/XTrainer/exp/exp1_neg_vanished/semantic_unrelated_pairs_199.json"

    # 加载三类句子对
    neg_pos_pairs = read_negpos_dataset(neg_path)
    sim_pairs = read_similar_dataset(sim_path)
    unrel_pairs = read_unrelated_dataset(unrel_path)

    # 获取每类的句子（均摊数量避免偏差）
    n = min(len(neg_pos_pairs), len(sim_pairs), len(unrel_pairs), 200)

    neg_sentences = [pair[0] for pair in neg_pos_pairs[:n]]
    pos_sentences = [pair[1] for pair in neg_pos_pairs[:n]]
    sim_sentences = [pair[0] for pair in sim_pairs[:n]] + [pair[1] for pair in sim_pairs[:n]]
    unrel_sentences = [pair[0] for pair in unrel_pairs[:n]] + [pair[1] for pair in unrel_pairs[:n]]

    all_sentences = neg_sentences + pos_sentences + sim_sentences + unrel_sentences
    print(f"总句子数量: {len(all_sentences)}")

    # 标签设定：
    # 0 - positive (肯定)
    # 1 - negative (否定)
    # 2 - similar (语义相似)
    # 3 - unrelated (语义无关)
    labels = (
        [1] * len(neg_sentences) +  # negative
        [0] * len(pos_sentences) +  # positive
        [2] * len(sim_sentences) +
        [3] * len(unrel_sentences)
    )
    labels = np.array(labels)

    print("正在提取所有句子的特征...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Clip_model.to(device)
    all_features = extract_clip_features(all_sentences)

    # 可视化（调用已有函数）
    for method in methods:
        print(f"正在使用 {method.upper()} 可视化特征...")
        visualize_multiclass_features(all_features, labels, method=method, title=f"CLIP Feature Visualization - {method.upper()}")

def visualize_multiclass_features(features, labels, method='pca', title=None):
    """
    可视化多类分布
    labels: 
      0 -> positive
      1 -> negative
      2 -> semantic similar
      3 -> semantic unrelated
    """
    n_samples = features.shape[0]
    print(f"样本数量: {n_samples}")
    perplexity = min(30, (n_samples - 1) // 3)  # 经验上除以 3 是个保守做法
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("method must be 'tsne' or 'pca'")

    print(f"正在使用 {method.upper()} 进行降维...")
    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    label_map = {
        0: ('Positive', 'blue'),
        1: ('Negative', 'red'),
        2: ('Semantic Similar', 'green'),
        3: ('Unrelated', 'gray')
    }

    for label_value, (label_name, color) in label_map.items():
         idx = labels == label_value
         plt.scatter(reduced[idx, 0], reduced[idx, 1], c=color, label=label_name, alpha=0.6)

    output_path = os.path.join(output_dir, f"{title}.png")

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
         
  # 启动可视化实验
if __name__ == "__main__":
   run_visualization_experiment(methods=['pca', 'tsne'])    