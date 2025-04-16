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

##### 实验1：二分类任务验证

​	训练分类器判断句子是否含否定词（二分类）。

* **数据集**：
  - **正样本**：所有否定句（如“没有猫的沙发”）。
  - **负样本**：所有肯定句（如“有猫的沙发”）。
  - **注意**：需确保正负样本的语义相似度匹配（避免分类器仅依赖语义差异）。
* 模型**：使用逻辑回归（Logistic Regression），输入为CLIP特征 `h`。
* **评估指标**：准确率（Accuracy）、F1分数。
  - 若准确率接近随机猜测（50%）：说明 `h` 中无否定信息。
  - 若准确率显著高于随机：说明 `h` 保留否定信息。

"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)
sys.path.insert(0, parent_dir)
from get_model import build_clip_model
from model.Clip import tokenize
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

config_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"

Clip_model = build_clip_model(config_path=config_path)

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
    return neg_pos_pairs

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

def run_binary_classification_experiment(path):
    neg_pos_pairs = read_negpos_dataset(path)
    
    # 二分类任务： 
    # - 正样本（类别1）：带有否定词的句子
    # - 负样本（类别0）：不带否定词的句子
    negative_sentences = [pair[0] for pair in neg_pos_pairs]
    positive_sentences = [pair[1] for pair in neg_pos_pairs]
    
    # 创建标签
    negative_labels = np.ones(len(negative_sentences))
    positive_labels = np.zeros(len(positive_sentences))
    
    # 合并所有句子和标签
    all_sentences = negative_sentences + positive_sentences
    all_labels = np.concatenate([negative_labels, positive_labels])
    
    print("正在提取CLIP特征...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Clip_model.to(device)
    all_features = extract_clip_features(all_sentences)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # 训练逻辑回归分类器
    print("正在训练分类器...")
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 评估模型
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    if accuracy > 0.6:  # 显著优于随机猜测
        print("CLIP特征包含否定信息！")
    else:
        print("CLIP特征似乎无法很好地捕捉否定信息。")
    
    return accuracy, f1

if __name__ == "__main__":
    path = "/root/NP-CLIP/XTrainer/exp/exp1_neg_vanished/neg_pos_pairs_166.json"
    run_binary_classification_experiment(path)
