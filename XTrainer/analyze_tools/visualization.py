"""
使用示例：

python -m analyze_tools.visualization

功能：
    1. 可视化注意力权重
    2. 可视化每一层每个区域的注意力
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import build_model
from utils import mkdir_if_missing, load_yaml_config
from transformers import CLIPTokenizer

def get_text_attention_map(model, text_tokens):
    """
    获取 CLIP 文本编码器每一层注意力层的注意力权重

    参数:
        - model (torch.nn.Module): 安装了钩子的文本编码模型。
        - text_tokens (torch.Tensor): 文本张量 | [batch_size, seq_len]

    返回:
        - attention_maps (np.ndarray): 注意力权重，形状为 (num_layers, num_heads, seq_len, seq_len)
    """
    attention_maps = []

    # 转移到模型的设备上
    text_tokens = text_tokens.unsqueeze(0).to(model.device)

    with torch.no_grad():
        print("正在进行前向传播...")
        # 将输入文本转换为 token 嵌入
        x = model.token_embedding(text_tokens).type(model.dtype)  # [num_classes, context_length, transformer_width]
        # 加上可训练的位置编码，保留序列位置信息
        x = x + model.positional_embedding.type(model.dtype)
        
        # 通过 Transformer 进行文本编码
        x = x.permute(1, 0, 2)  # 调整维度为 [context_length, num_classes, transformer_width] 以适配 Transformer
        x = model.transformer(x, attention_maps=attention_maps)
        x = x.permute(1, 0, 2)  # 还原维度为 [num_classes, context_length, transformer_width]

        # 通过 layerNorm 层归一化数据
        x = model.ln_final(x).type(model.dtype)

        # 使用 EOT (End-of-Text) token 对应的特征作为整个文本序列的表示 (类似 Bert 用 [cls] token)
        EOT = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)]  
        print(f"前向传播完成！获取到{len(attention_maps)} 层的注意力权重")
        attention_maps = np.array([attn.detach().cpu().numpy() for attn in attention_maps])

    return attention_maps


def get_img_attention_map(model, img):
    """
    获取 Clip 图像编码器每一层注意力层的的注意力权重
    
    参数:
        - model (torch.nn.Module): 安装了钩子的注意力模型。
        - img (torch.Tensor): 图像张量 | [batch_size, 3, input_size, input_size]
        
    返回:
        - attention_maps (np.ndarray): 注意力权重，形状为 (num_layers, num_heads, height, width)。
    """

    # 1. 获取 model 的每一层的注意力权重
    attention_maps = []
    with torch.no_grad():  # 关闭梯度计算
       model.visual(img, attention_maps) # attention_maps 是放置在模型中的一个钩子，用于存储每一层的注意力权重
    
    # 2. 转换注意力权重为 NumPy 数组
    attention_maps = np.array([attn.detach().cpu().numpy() for attn in attention_maps])
    return attention_maps


def display_attn_weight(model, img, visualize=False, output_path=None):
    """
    可视化注意力权重

    参数:
        - model (torch.nn.Module): ViT 等包含注意力层的模型。
        - img（torch.Tensor）: 图像张量 | [batch_size, 3, input_size, input_size]
    """
    # 3. 计算注意力映射
    attention_maps = get_img_attention_map(model, img)

    if not visualize:
        return attention_maps
    else:
        # 5. 可视化注意力映射
        plt.figure(figsize=(12, 6))
        num_layers = min(20, len(attention_maps))  # 只画最多 6 层

        for i in range(num_layers):
            plt.subplot(num_layers//4, 4, i + 1)
            plt.imshow(attention_maps[i].squeeze(0), cmap="Reds")
            plt.title(f"Layer {i+1}")
            plt.colorbar()  # 添加颜色条
            plt.axis("off")

        plt.tight_layout()  # 自动调整子图布局
        plt.savefig(output_path)

        return attention_maps  # 返回注意力权重



# 可视化图像每一层每个区域的注意力
def display_img_attn(model, img, visualize=False, output_path=None):
    """
    可视化每一层每个区域的注意力

    参数:
        - model (torch.nn.Module): ViT 等包含注意力层的模型。
        - img（torch.Tensor）: 图像张量 | [batch_size, 3, input_size, input_size]
    """
    # 1. 获取模型的每一层的注意力权重
    attention_maps = get_img_attention_map(model, img)
    attention_maps = np.expand_dims(attention_maps[:, :, 0, 1:], axis=2)  # 只观察 cls token 关注哪些像素  (12, 1, 1, 49) 
    
    # 2. 计算每个 patch 受到的总注意力
    # 对每一层的每列求和，得到每个 patch 受到的关注度
    num_layers = len(attention_maps)
    layer_attentions = []
    for i in range(num_layers):
        layer_attention = np.sum(attention_maps[i][0], axis=0)  # 取第 i 层的注意力权重 (1, 50, 49)->(1, 49) 
        layer_attention = np.clip(layer_attention.astype(np.float32),0,1)
        # 归一化
        layer_attention = (layer_attention - layer_attention.min()) / (layer_attention.max() - layer_attention.min())
        layer_attentions.append(layer_attention)

    # 3. 计算所有层的注意力热图叠加 并 归一化
    total_attention = np.sum(layer_attentions, axis=0)
    total_attention = (total_attention - total_attention.min()) / (total_attention.max() - total_attention.min())

    # 4. 重新映射回 224x224 图像
    patchsize = model.visual.patch_size
    size = int(np.sqrt(224*224 // (patchsize*patchsize)))  # 224*224/16*16 = 14*14
    heatmaps = []
    for layer_attention in layer_attentions:
        heatmap = cv2.resize(layer_attention.reshape(size, size), (224, 224), interpolation=cv2.INTER_CUBIC) # ViTb16: 224*224/16*16 = 14*14
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 重新归一化到 [0, 1] | 插值后会导致数值范围变化
        heatmaps.append(heatmap)
    total_heatmap = cv2.resize(total_attention.reshape(size, size), (224, 224), interpolation=cv2.INTER_CUBIC)

    # 5. 可视化原图 + 热图
    if not visualize:
        return heatmaps, total_heatmap
    else:
        plt.figure(figsize=(15, 15))

        # 显示原图
        image_for_display = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_for_display = np.clip(image_for_display.astype(np.float32),0,1) # 转为 float32 类型并归一化到 [0, 1]
        # image_for_display = (image_for_display - image_for_display.min()) / (image_for_display.max() - image_for_display.min())
        plt.subplot(4, 4, 1)  # 在 4x4 网格的第一个位置显示原图
        plt.imshow(image_for_display)
        plt.axis("off")
        plt.title("Original Image")

        # 显示每一层的热图
        for i, heatmap in enumerate(heatmaps):
            plt.subplot(4, 4, i + 2)  # 在 4x4 网格的后续位置显示每一层的热图
            plt.imshow(image_for_display, alpha=0.5)
            plt.imshow(heatmap, cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.axis("off")
            plt.title(f"Attention Heatmap Layer {i+1}")

        # 显示总热图（所有层叠加）
        plt.subplot(4, 4, num_layers + 2)  # 在 4x4 网格的最后一个位置显示总热图
        plt.imshow(image_for_display, alpha=0.5)
        plt.imshow(total_heatmap, cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.axis("off")
        plt.title("Total Attention Heatmap (Sum of All Layers)")

        plt.tight_layout()
        mkdir_if_missing(output_path)
        print(f"保存 attention heatmaps 至 {output_path}/img_attn.png")
        plt.savefig(f"{output_path}/img_attn.png")


def attention_rollout(attention_maps, residual=True):
    """
    类似于 Attention Rollout 或者 Attention Flow 的思想
    将每一层的注意力权重进行累乘传播，得到最终的注意力权重

    参数:
        - attention_maps (torch.Tensor): 注意力权重，形状为 (num_layers=12, num_heads=1, T, T)
        - residual (bool): 是否使用残差连接
        - device (str): 设备类型，默认为 'cpu'
        
    返回:
        - result (torch.Tensor): 最终的注意力权重，形状为 (T, T) | T: token 数量
    """
    num_layers, num_heads, T, _ = attention_maps.shape
    result = np.eye(T)

    for i in range(num_layers):
        attn = attention_maps[i]  # [H, T, T]
        attn_mean = attn.mean(axis=0)  # [T, T]

        if residual:
            attn_resid = 0.5 * attn_mean + 0.5 * np.eye(T)
        else:
            attn_resid = attn_mean

        result = attn_resid @ result  # 累乘传播
    print("attention_rollout 完成！")
    return result  # [T, T]


def display_tokens_contribution(model, text, visualize=False, output_path=None):
    """
    可视化输入文本中每个 token 对最终 EOT 特征的贡献度

    参数:
        - model: 模型对象
        - text: 文本输入
        - visualize: 是否可视化
        - output_path: 可视化图像的保存路径
    """
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    tokens = inputs.input_ids[0]
    attention_maps = get_text_attention_map(model, tokens)  # [L, H, T, T]

    rollout = attention_rollout(attention_maps)  # [T, T]

    eos_idx = (tokens != tokenizer.pad_token_id).nonzero()[-1].item()
    sos_idx = (tokens != tokenizer.pad_token_id).nonzero()[0].item()

    # 取出 SOS 到 EOS 之间的 token 对 EOS 的总贡献度
    meaningful_tokens = tokens[sos_idx + 1:eos_idx + 1]
    meaningful_words = tokenizer.convert_ids_to_tokens(meaningful_tokens)
    eos_contribution = rollout[eos_idx][sos_idx + 1:eos_idx + 1]  # shape: [有效 token 数]

    if not visualize:
        return eos_contribution, meaningful_words

    plt.figure(figsize=(15, 2.5))
    plt.bar(range(len(meaningful_tokens)), eos_contribution)
    plt.xticks(range(len(meaningful_tokens)), meaningful_words, rotation=90)
    plt.title("Token Contributions to Final EOS Feature (Attention Rollout)")
    plt.tight_layout()

    if output_path:
        mkdir_if_missing(output_path)
        plt.savefig(f"{output_path}/{text}.png")
        print(f"图像保存至 {output_path}/{text}.png")

    return eos_contribution, meaningful_words

def display_word_attn(model, text, word="not", visualize=False, output_path=None):
    """
    可视化输入文本中的单词word对其他所有token的注意力权重
    
    参数:
        - model: 模型对象
        - text: 文本输入
        - word(str): 单词
        - visualize: 是否可视化
        - output_path: 可视化图像的保存路径
        
    返回:
        - attention_to_meaningful: 注意力权重
        - meaningful_words: 有意义的token
        - word: 单词
    """
    # 使用tokenizer处理输入文本
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=77
    )
    tokens = inputs.input_ids[0]
    attention_maps = get_text_attention_map(model, tokens)  # [L, H, T, T]
    
    # 转换tokens为可读文本并找到否定词的索引
    token_words = tokenizer.convert_ids_to_tokens(tokens)
    
    # 找到否定词的索引
    w_indices = [i for i, w in enumerate(token_words) if word in w]
    if not w_indices:
        print(f"没有找到单词 '{word}' 在文本中")
        return None, None, None
    
    w_idx = w_indices[0]  # 使用第一个出现的word
    
    # 计算否定词的注意力权重
    rollout = attention_rollout(attention_maps)  # [T, T]
    
    # 找到有意义的token范围（排除padding）
    eos_idx = (tokens != tokenizer.pad_token_id).nonzero()[-1].item()
    sos_idx = (tokens != tokenizer.pad_token_id).nonzero()[0].item()
    
    # 取出有意义范围内的token和对应的注意力权重
    meaningful_tokens = tokens[sos_idx+1: eos_idx+1]
    meaningful_words = tokenizer.convert_ids_to_tokens(meaningful_tokens)
    attention_to_meaningful = rollout[w_idx][sos_idx+1: eos_idx+1]
    
    if not visualize:
        return attention_to_meaningful, meaningful_words, word

    plt.figure(figsize=(15, 2.5))
    plt.bar(range(len(meaningful_words)), attention_to_meaningful)
    plt.xticks(range(len(meaningful_words)), meaningful_words, rotation=90)
    plt.title(f"Attention from Word '{word}' to Other Tokens")
    plt.tight_layout()
    
    if output_path:
        mkdir_if_missing(output_path)
        plt.savefig(f"{output_path}/attn_{word}_{text}.png")
        print(f"图像保存至 {output_path}/attn_{word}_{text}.png")
    
    return attention_to_meaningful, meaningful_words, word
    

if __name__ == "__main__":
    cfg_path = "/root/NP-CLIP/XTrainer/config/CLS/CLS-Clip-VitB16-ep50-Caltech101-SGD.yaml"
    
    """可视化图像注意力"""
    # image_path = "/root/NP-CLIP/X-Trainer/analyze_tools/imgs/white.jpg"  # 图像路径
    # output_path = "/root/NP-CLIP/X-Trainer/analyze_tools/output"  # 输出路径
    # cfg = load_yaml_config(cfg_path) # 读取配置

    # # 加载模型
    # model = build_model(cfg)  

    # # 加载图像，转为张量[batch_size, 3, input_size, input_size]
    # input_size = cfg.INPUT.SIZE  # 输入大小
    # intermode = cfg.INPUT.INTERPOLATION  # 插值模式
    # img = Image.open(image_path)  # 打开图像
    # transform = Compose([standard_image_transform(input_size, intermode), ToTensor()])  # 定义转换
    # img = transform(img)  # 转换图像
    # img = img.unsqueeze(0)  # 添加 batch 维度
    # img = img.to(dtype=model.dtype, device=model.device)  # 转移到模型的设备上

    # # 可视化注意力权重
    # display_img_attn(model, img, visualize=True, output_path=output_path)  # 可视化注意力权重

    """可视化文本注意力"""
    text = "There is not a dog in the house"
    output_path = "output"  # 输出路径
    cfg = load_yaml_config(cfg_path) # 读取配置
    # 加载模型
    model = build_model(cfg)  
    # 可视化文本注意力
    attention_to_meaningful, meaningful_words = display_tokens_contribution(model, text, visualize=True, output_path=output_path)  # 可视化文本注意力
    print(f"EOS \n - attention_to_meaningful = {attention_to_meaningful} \n - meaningful_words = {meaningful_words}")
    # attention_to_meaningful, meaningful_words, word = display_word_attn(model, text, word="not", visualize=True, output_path=output_path)  # 可视化文本注意力
    # print(f"word = {word} \n - attention_to_meaningful = {attention_to_meaningful} \n - meaningful_words = {meaningful_words}")
    # attention_to_meaningful, meaningful_words, word = display_word_attn(model, text, word="dog", visualize=True, output_path=output_path)  # 可视化文本注意力
    # print(f"word = {word} \n - attention_to_meaningful = {attention_to_meaningful} \n - meaningful_words = {meaningful_words}")
    # attention_to_meaningful, meaningful_words, word = display_word_attn(model, text, word="house", visualize=True, output_path=output_path)  # 可视化文本注意力
    # print(f"word = {word} \n - attention_to_meaningful = {attention_to_meaningful} \n - meaningful_words = {meaningful_words}")
    