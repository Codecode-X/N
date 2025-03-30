"""
这段代码的主要作用是：
- 加载 CLIP 模型到 CPU：如果模型是 JIT 格式，直接加载；否则加载 State Dict。
- 读取 Prompt 文件：提取学习到的上下文向量。
- 计算相似度：基于上下文向量与 Token 嵌入矩阵计算欧几里得距离。
- 输出最相似的单词：返回与上下文最相似的 top-k 单词及其距离，用于分析 Prompt 的有效性。

用途场景：
- 分析 Prompt 学习的质量。
- 理解模型的 Token 表示。
- 进行 Prompt 工程调试。

"""

import os
import sys
import argparse
import torch

from clip.simple_tokenizer import SimpleTokenizer
from clip import clip


def load_clip_to_cpu(backbone_name="RN50"):
    """加载 CLIP 模型到 CPU。
    参数：
        backbone_name (str): CLIP 模型的骨干网络名称。
    返回：
        torch.nn.Module: 加载的 CLIP 模型。
    """
    url = clip._MODELS[backbone_name]  # 获取模型的下载 URL
    model_path = clip._download(url)  # 下载模型文件

    try:
        # 尝试加载 JIT 格式的模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None  # 如果成功加载 JIT 模型，则不需要 state_dict

    except RuntimeError:
        # 如果 JIT 加载失败，则加载 state_dict 格式的模型
        state_dict = torch.load(model_path, map_location="cpu")

    # 构建 CLIP 模型
    model = clip.build_model(state_dict or model.state_dict())

    return model  # 返回加载的模型

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
# 添加命令行参数：fpath 表示学习到的 prompt 文件路径
parser.add_argument("fpath", type=str, help="Path to the learned prompt")
# 添加命令行参数：topk 表示选择最相似的 top-k 个单词
parser.add_argument("topk", type=int, help="Select top-k similar words")
args = parser.parse_args()  # 解析命令行参数

fpath = args.fpath  # 获取 prompt 文件路径
topk = args.topk  # 获取 top-k 值

# 确保提供的文件路径存在
assert os.path.exists(fpath)

print(f"Return the top-{topk} matched words")  # 打印提示信息

tokenizer = SimpleTokenizer()  # 初始化简单分词器
clip_model = load_clip_to_cpu()  # 加载 CLIP 模型到 CPU
token_embedding = clip_model.token_embedding.weight  # 获取 token 的嵌入权重
print(f"Size of token embedding: {token_embedding.shape}")  # 打印 token 嵌入的大小

# 加载学习到的 prompt 的状态字典
prompt_learner = torch.load(fpath, map_location="cpu")["state_dict"]
ctx = prompt_learner["ctx"]  # 提取上下文向量
ctx = ctx.float()  # 转换为浮点类型
print(f"Size of context: {ctx.shape}")  # 打印上下文向量的大小

if ctx.dim() == 2:
    # 如果上下文是二维的，表示通用上下文
    distance = torch.cdist(ctx, token_embedding)  # 计算上下文与 token 嵌入之间的距离
    print(f"Size of distance matrix: {distance.shape}")  # 打印距离矩阵的大小
    sorted_idxs = torch.argsort(distance, dim=1)  # 按距离排序
    sorted_idxs = sorted_idxs[:, :topk]  # 取前 top-k 个索引

    for m, idxs in enumerate(sorted_idxs):
        # 遍历每个上下文，获取对应的 top-k 单词和距离
        words = [tokenizer.decoder[idx.item()] for idx in idxs]  # 解码单词
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]  # 格式化距离
        print(f"{m+1}: {words} {dist}")  # 打印结果

elif ctx.dim() == 3:
    # 如果上下文是三维的，表示类特定上下文
    raise NotImplementedError  # 尚未实现
