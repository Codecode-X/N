import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 可视化注意力权重
def display_attn_weight(image_path):
    # 1. 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # 设置为评估模式，避免 Dropout 或 BatchNorm 影响

    # 2. 读取图像并预处理
    image_path = "clip/raw.png"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 3. 获取 ViT 的注意力层
    def get_attention_map(model, image):
        vit_model = model.visual  # CLIP 的 ViT 图像编码器
        attention_maps = []
        with torch.no_grad():  # 关闭梯度计算
            _ = vit_model(image, attention_maps)
        # 转换注意力权重为 NumPy 数组
        attention_maps_np = [attn.detach().cpu().numpy() for attn in attention_maps]
        return np.array(attention_maps_np)

    # 4. 计算注意力映射
    attention_maps = get_attention_map(model, image)

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
    plt.savefig("exp/attn_weight.png")



# 可视化图像每一层每个区域的注意力
def display_img_attn(image_path):
    # 1. 加载 CLIP 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()  # 评估模式，避免 Dropout 影响
    # 2. 读取并预处理图像
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 3. 获取 ViT 的注意力层
    def get_attention_map(model, image):
        vit_model = model.visual  # CLIP 的 ViT 图像编码器
        attention_maps = []
        with torch.no_grad():  # 关闭梯度计算
            _ = vit_model(image, attention_maps)
        # 转换注意力权重为 NumPy 数组
        attention_maps_np = [attn.detach().cpu().numpy() for attn in attention_maps]
        return np.array(attention_maps_np)

    # 4. 计算注意力映射
    attention_maps = get_attention_map(model, image)  # 形状 (12, 1, 50, 50) 包含 cls token
    # attention_maps = attention_maps[:, :, :, 1:]  # 观察全局互注意力 - 取索引 1: 之后的部分，去除 cls token，形状 (12, 1, 50, 49) 
    
    attention_maps = np.expand_dims(attention_maps[:, :, 0, 1:], axis=2)  # 只观察 cls token 关注哪些像素  (12, 1, 1, 49) 
    
    # 5. 计算每个 patch 受到的总注意力
    # 对每一层的每列求和，得到每个 patch 受到的关注度
    num_layers = len(attention_maps)
    layer_attentions = []
    for i in range(num_layers):
        layer_attention = np.sum(attention_maps[i][0], axis=0)  # 取第 i 层的注意力权重 (1, 50, 49)->(1, 49) 
        # 归一化
        layer_attention = (layer_attention - layer_attention.min()) / (layer_attention.max() - layer_attention.min())
        layer_attentions.append(layer_attention)

    # 6. 计算所有层的注意力热图叠加 并 归一化
    total_attention = np.sum(layer_attentions, axis=0)
    total_attention = (total_attention - total_attention.min()) / (total_attention.max() - total_attention.min())

    # 7. 重新映射回 224x224 图像
    heatmaps = []
    for layer_attention in layer_attentions:
        heatmap = cv2.resize(layer_attention.reshape(7, 7), (224, 224), interpolation=cv2.INTER_CUBIC)
        heatmaps.append(heatmap)
    total_heatmap = cv2.resize(total_attention.reshape(7, 7), (224, 224), interpolation=cv2.INTER_CUBIC)

    # 8. 可视化原图 + 热图
    plt.figure(figsize=(15, 15))

    # 显示原图
    image_for_display = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_for_display = np.clip(image_for_display, 0, 1)

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
    plt.savefig("exp/img_attn.png")