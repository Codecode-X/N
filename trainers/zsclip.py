import torch
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from clip import clip
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES_SELECT

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    """ 
    使用 CLIP 进行零样本学习（单一提示模板-CUSTOM_TEMPLATES）。
    """
    def build_model(self):
        """ 构建模型。"""
        cfg = self.cfg  # 获取配置
        classnames = self.dm.dataset.classnames  # 获取数据集的类别名称

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")  # 打印加载的 CLIP 模型信息
        clip_model = load_clip_to_cpu(cfg)  # 加载 CLIP 模型到 CPU
        clip_model.to(self.device)  # 将模型移动到指定设备（如 GPU）

        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]  # 获取当前数据集的模板
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]  # 根据类别名称生成提示
        print(f"Prompts: {prompts}")  # 打印生成的提示
        prompts = torch.cat([clip.tokenize(p) for p in prompts])  # 对提示进行 tokenize
        prompts = prompts.to(self.device)  # 将 tokenized 提示移动到指定设备

        with torch.no_grad():  # 禁用梯度计算
            text_features = clip_model.encode_text(prompts)  # 编码文本提示
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 对特征进行归一化

        self.text_features = text_features  # 保存文本特征
        self.clip_model = clip_model  # 保存 CLIP 模型

    def model_inference(self, image):
        """ 模型推理。"""
        image_features = self.clip_model.encode_image(image)  # 编码图像
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 对图像特征进行归一化
        logit_scale = self.clip_model.logit_scale.exp()  # 获取 logit 缩放因子
        logits = logit_scale * image_features @ self.text_features.t()  # 计算图像与文本特征的相似度
        return logits  # 返回相似度


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """
    使用 CLIP 进行零样本学习（多个提示模板求平均）。
    (继承自 ZeroshotCLIP。)
    """
    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT  # 默认 IMAGENET 提示模板

    def build_model(self):
        """ 构建模型。
        1. 加载 CLIP 模型。
        2. 生成多个提示模板。
        3. 计算平均文本特征。
        """
        cfg = self.cfg  # 获取配置
        classnames = self.dm.dataset.classnames  # 获取数据集的类别名称

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")  # 打印加载的 CLIP 模型信息
        clip_model = load_clip_to_cpu(cfg)  # 加载 CLIP 模型到 CPU
        clip_model.to(self.device)  # 将模型移动到指定设备（如 GPU）

        for params in clip_model.parameters():  # 冻结 CLIP 模型的所有参数
            params.requires_grad_(False)

        # 添加自定义的提示模板
        if cfg.DATASET.NAME != "ImageNet":  # 如果数据集不是 ImageNet 则在默认 IMAGENET 模板基础上添加自定义模板
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]
        else:
            pass # 如果是 ImageNet 则直接使用默认 IMAGENET 提示模板

        num_temp = len(self.templates)  # 获取模板数量
        print(f"Prompt ensembling (n={num_temp})") 

        # 计算所有模板的平均文本特征
        mean_text_features = 0  # 平均文本特征
        for i, temp in enumerate(self.templates):  # 遍历所有模板
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]  # 根据类别名称生成提示
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)  # 对提示进行 tokenize 并移动到设备
            text_features = clip_model.encode_text(prompts)  # 编码文本提示
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 对特征进行归一化
            mean_text_features = mean_text_features + text_features  # 累加特征
        mean_text_features = mean_text_features / num_temp  # 计算平均特征
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)  # 对平均特征进行归一化

        self.text_features = mean_text_features  # 保存平均文本特征
        self.clip_model = clip_model  # 保存 CLIP 模型
