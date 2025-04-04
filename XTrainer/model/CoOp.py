from .Clip import Clip, tokenize
from torch import nn
import torch
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

@MODEL_REGISTRY.register()
class CoOp(ModelBase):
    """ 
    CoOp 模型：基于提示学习调优的图像和文本的对比学习
    该模型使用 CLIP 模型作为基础，并在其上添加了一个提示学习器（Prompt Learner），用于生成每个类别的最优提示信息。
    
    主要步骤：
    1. 编码图像特征
    2. 利用可学习的 Prompt Learner 生成没饿过类别的文本提示词，并编码文本特征
    3. 计算图像和文本之间的余弦相似度
    4. 返回与图像最相似的文本提示词对应的类别作为结果
    """
    def __init__(self, cfg):
        """
        初始化 CoOp 模型
        
        参数：
        cfg: 配置文件，包含模型的超参数和训练设置
        """
        super().__init__(cfg)  # 调用父类 Clip 的构造函数
        # 读取预训练的 CLIP 模型
        pretrained_clip = Clip.build_model(cfg) # 调用父类 build 得到预训练的 CLIP 模型 (读取预训练权重)
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":  # clip模型默认是 float16 精度
            pretrained_clip.float() # 将模型转换为 float32 精度
            
        self.image_encoder = pretrained_clip.visual  # 图像编码器
        self.text_encoder = pretrained_clip.transformer  # 文本编码器
        self.token_embedding = pretrained_clip.token_embedding  # 词嵌入层
        self.positional_embedding = pretrained_clip.positional_embedding  # 位置嵌入层
        self.ln_final = pretrained_clip.ln_final  # 最终的 LayerNorm 层
        self.text_projection = pretrained_clip.text_projection  # 文本投影层
        self.logit_scale = pretrained_clip.logit_scale  # 温度参数 τ 的倒数
        self.device = pretrained_clip.device  # 设备
        self.dtype = pretrained_clip.dtype  # 数据类型
        self.cfg = cfg  # 配置文件

        # 提示学习 - 通过训练器调用 init_promptLearner 初始化
        self.pptLearner = None  # 提示学习器
        self.eot_indices = None  # 每个类别的 prompt 的结束符位置

    def init_promptLearner(self, pptLearner, cls_list:list, task_mode:str):
        """
        初始化提示学习器
        
        两种使用场景：
            - 分类任务：在训练开始前，提前定义所有类别的文本标签，初始化提示学习器
            - MCQ任务：在每次前向传播前，动态生成每个样本的文本标签，初始化提示学习器
        
        参数：
            - pptLearner (PromptLearner): 提示学习器对象
            - cls_list (list): 类别/选项文本列表 | [num_classes] | ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - task_mode (str): 任务模式，"CLS" 或 "MCQ"
        
        主要步骤：
            1. 初始化一个 PromptLearner 对象 self.pptLearner，用于学习提示信息
            2. 获取每个类别的 prompt 的结束符位置 eot_indices，保存至 self.eot_indices
        """
        print("正在初始化 PromptLearner...")

        """获取通用可学习的上下文向量 ctx"""
        self.pptLearner = pptLearner  # 提示学习器对象
        n_ctx = self.pptLearner.n_ctx  # 上下文词数量
        ctx = self.pptLearner.ctx  # 获取通用可学习的上下文向量 ctx | shape: (n_ctx, dim)

        """构造 每个类别的 prompt ([SOS] + 上下文向量 ctx(可学习) + 类别名 + 句号 + [EOS])"""
        tokenized_prompts = []
        eot_indices = []
        for cls in cls_list:
            prompt, eot_indice = self.pptLearner.construct_prompt(ctx, cls)
            tokenized_prompts.append(tokenize(prompt))
            eot_indices.append(eot_indice)

        tokenized_prompts = torch.cat(tokenized_prompts, dim=0).to(self.device)  # 所有类别的 tokenized prompt | shape: (n_cls, context_len)
        eot_indices = torch.tensor(eot_indices, dtype=torch.long, device=self.device)  # 每个类别的 prompt 的结束符位置 | shape: (n_cls,)

        with torch.no_grad():
            ctx_embedding = self.token_embedding(tokenized_prompts).type(self.dtype)  # 得到 tokenized_prompts 的嵌入表示 | shape: (n_cls, context_len, dim)
        
        # 得到 pptLearner 的 token_prefix 和 token_suffix 
        if task_mode == "CLS":
            # 静态注册buffer（持久化保存），节省每次重复计算带来的开销
            self.pptLearner.register_buffer("token_prefix", ctx_embedding[:, :1, :]) # [SOS] token | shape: (n_cls, 1, dim)
            self.pptLearner.register_buffer("token_suffix", ctx_embedding[:, 1+n_ctx:, :]) # 类别名 + [EOS] token | shape: (n_cls, *, dim)
        elif task_mode == "MCQ":
            # 动态存储为实例属性（非持久化），每次前向传播时计算
            self.pptLearner.token_prefix = ctx_embedding[:, :1, :].to(self.device) # [SOS] token | shape: (n_cls, 1, dim)
            self.pptLearner.token_suffix = ctx_embedding[:, 1+n_ctx:, :].to(self.device) # 类别名 + [EOS] token | shape: (n_cls, *, dim)
        else:
            raise ValueError(f"未知任务模式: {task_mode}")

        self.eot_indices = eot_indices  # 每个类别的 prompt 的结束符位置

    def forward(self, image, return_feature=False):
        """ 
        重写 forward 函数 
        
        参数：
            - image(torch.Tensor): 输入图像，形状为 (batch_size, 3, height, width)
            - return_feature(bool): 是否返回图像特征，默认为 False

        主要步骤：
            1. 编码图像特征
            2. 生成每个类别的 prompt，拼接可训练的 prompts 和 可训练的位置嵌入 得到文本输入
            3. 编码文本特征
            4. 计算图像和文本之间的余弦相似度
            5. 返回与图像最相似的文本提示词对应的类别作为结果
        """
        # ---编码图像特征---
        image_features = self.image_encoder(image.type(self.dtype))  
        # ---生成 prompt---
        prompts = self.pptLearner() # 获取学习到的每个类别的 prompt
        # ---编码文本特征---
        # 可训练的 prompts + 可训练的位置嵌入
        t = prompts + self.positional_embedding.type(self.dtype)
        # 交换维度，使其符合文本编码器输入格式 
        t = t.permute(1, 0, 2)  # (batch_size, n_ctx, width) -> (n_ctx, batch_size, width) | n_ctx(序列长度=num_patches+1) width(隐藏层宽度)
        t = self.text_encoder(t) # 编码文本特征
        t = t.permute(1, 0, 2)  # LND -> NLD
        t = self.ln_final(t).type(self.dtype) # (n_ctx, batch_size, width) -> (batch_size, n_ctx, width)
        # 获取文本编码特征
        batch_indices = torch.arange(t.shape[0])
        EOT = t[batch_indices, self.eot_indices]
        # 线性投影得到文本特征
        text_features = EOT @ self.text_projection
        # ---计算 图像 和 文本 间的 余弦相似度---
        logit_scale = self.logit_scale.exp() # 温度参数 τ 的倒数
        logits_per_image = logit_scale * image_features @ text_features.t() # image->text 相似度 | [batch, num_classes]
        # ---返回结果---
        if return_feature: 
            return logits_per_image, image_features
        else:
            return logits_per_image
        

class PromptLearner(nn.Module):
    """ 
    提示学习器：用学习所有类别通用的上下文词ctx来生成每个类别的提示信息
    
    参数：
        - cfg: 配置文件，包含模型的超参数和训练设置
        - classnames: 类别名称列表，用于生成提示信息
        - clip_model: 实例化的 CLIP 模型对象

    配置：
        - cfg.MODEL.init_ctx: 初始化的上下文词，例如 "a photo of a"
    """
    def __init__(self, cfg, clip_model):
        """
        初始化提示学习器

        参数：
            - cfg: 配置文件，包含模型的超参数和训练设置
            - classnames: 类别名称列表，用于生成提示信息 | 如 ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - clip_model: 实例化的 CLIP 模型对象
        
        主要步骤：
        1. 读取参数和配置
        2. 初始化上下文前缀词和嵌入向量 | 上下文前缀词为所有类别通用
        3. 构造每个类别的完整的提示文本 -> [SOS] + 上下文向量 ctx(可学习) + 类别名 + 句号 + [EOS]
        4. 注册需持久化保存的张量 token_prefix 和 token_suffix
        5. 将上下文向量 learnable_ctx 设为可训练参数
        6. 记录每一类的提示文本的结尾 EOT 索引位置 eot_indices
        """
        super().__init__()
        # 读取参数
        dtype = clip_model.dtype  # CLIP 模型的数据类型
        clip_imsize = clip_model.image_encoder.input_resolution # CLIP 模型的输入图像尺寸
        # 读取配置
        ctx_init = cfg.MODEL.init_ctx  # 是否使用预设的上下文词 | 如 "a photo of a" | 所有类别通用的上下文词
        cfg_imsize = cfg.INPUT.SIZE # 配置文件中设定的输入图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) 必须等于 clip_imsize ({clip_imsize})" # 确保输入尺寸匹配

        """初始化 上下文前缀词 以及其 嵌入向量"""
        ctx_init = ctx_init.replace("_", " ")  # 将下划线替换为空格
        self.n_ctx = len(ctx_init.split(" "))  # 重新计算上下文词数量
        tokenized_ctx = tokenize(ctx_init)  # 将文字形式的 ctx_init 分词编码为 token | shape: (1, context_len=77)
        with torch.no_grad():
            tokenized_ctx = tokenized_ctx.to(clip_model.device)
            ctx_embedding = clip_model.token_embedding(tokenized_ctx).type(dtype)  # 获取嵌入表示 (1, context_len, embedding_dim)
        ctx_vectors = ctx_embedding[0, 1:1+self.n_ctx, :]  # 提取上下文嵌入向量 | 1:1+n_ctx: 索引 0 位置对应 [SOS] | shape: (n_ctx, dim)
        print(f'初始上下文向量："{ctx_init}"')
        print(f"上下文 token 数为 (tokens): {self.n_ctx}")
        
        # 将 所有类别通用的 上下文向量 ctx 设为 可训练参数
        self.ctx = nn.Parameter(ctx_vectors) # shape: (n_ctx, dim)

        # 待初始化参数 | 需要用 init_promptLearner 初始化赋值
        self.token_prefix = None  # [SOS] token | shape: (n_cls, 1, dim)
        self.token_suffix = None  # 类别名 + [EOS] token | shape: (n_cls, *, dim)

    def construct_prompt(self, ctx, cls):
        """
        构造 prompt 文本 

        构造的 prompt 文本格式为：[SOS] + 上下文 + 类别名 + 句号 + [EOS]
        例如：ctx_init = "a photo of a" + " " + cls + "." + [EOS]，其中 [EOS] 是结束符
        
        参数：
            - ctx (str): 上下文文本，例如 "a photo of a"
            - cls (str): 类别名称，例如 "sleeping_dog"

        返回：
            - prompt (str): 完整的 prompt 文本，例如 "a photo of a sleeping dog."
            - eot_indice (int): 每个类别的 prompt 的结束符位置索引
        """
        cls = cls.replace("_", " ")  # 处理类别/选项名称中的"_"
        prompt = ctx + " " + cls + "." # 生成 prompt 文本（ctx_init + ' ' + 类别/choice + 句号）
        tokenized_prompt = tokenize(prompt)  # 将文字形式的 prompt 分词编码为 token | shape: (1, context_len=77)
        eot_indice = tokenized_prompt.argmax(dim=-1)  # 获取 tokenized_prompt 的 [EOS] token 的 索引 | shape: (1,)
        return prompt, eot_indice

    def forward(self):
        learnable_ctx = self.ctx  # 取出可学习(通用)上下文向量 | shape: (n_ctx, dim)
        learnable_ctxs = learnable_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 扩展维度 | shape: (n_cls, n_ctx, dim)

        # 每个类别的 prompt 的 prefix 和 suffix
        prefixs = self.token_prefix  # [SOS]
        suffixs = self.token_suffix  # 包括 类别名 和 [EOS]
        
        # 类别名称放在结尾（论文实验显示"end"效果最好）
        prompts = torch.cat(
            [
                prefixs,  # (n_cls, 1, dim) [SOS] | shape: (n_cls, 1, dim)
                learnable_ctxs,  # (n_cls, n_ctx, dim) 上下文 | shape: (n_cls, n_ctx, dim)
                suffixs,  # (n_cls, *, dim) 包括 类别名+[EOS] | shape: (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts  # [SOS] + " " + 上下文 + 类别名 + "." + [EOS] | shape: (n_cls, *, dim)