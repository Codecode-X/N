from .Clip import Clip, tokenize
from utils import SimpleTokenizer as Tokenizer
from torch import nn
import torch.nn.functional as F
import torch
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase
import warnings

_tokenizer = Tokenizer() # 初始化分词器

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
        self._pptLearner = None  # 提示学习器

    def register_promptLearner(self, pptLearner):
        """
        注册提示学习器
        
        参数：
            pptLearner (PromptLearner): 提示学习器对象
        """
        self._pptLearner = pptLearner

    # ===============并行化================
    
    def batch_init_prompt_learner(self, batch_choices):
        """
        批量初始化提示学习器
        """
        # 展平所有选项 [batch_size * num_choices]
        all_classes = [cls for choices in batch_choices for cls in choices]  # 128 = 4 * 32
        
        # 批量生成后缀
        suffixs, eos_indices = self._pptLearner.batch_construct_suffix(all_classes, self)
        
        # 重组为 [batch_size, num_choices, ...]
        num_choices = len(batch_choices[0])
        self._pptLearner.suffixs = suffixs.view(len(batch_choices), num_choices, *suffixs.shape[1:])
        self.eos_indices = eos_indices.view(len(batch_choices), num_choices)


    # ====================================


    def init_promptLearner(self, cls_list:list, task_mode:str, pptLearner=None):
        """
        初始化提示学习器
        
        两种使用场景：
            - 分类任务：在训练开始前，提前定义所有类别的文本标签，初始化提示学习器
            - MCQ任务：在每次前向传播前，动态生成每个样本的文本标签，初始化提示学习器
        
        参数：
            - cls_list (list): 类别/选项文本列表 | [num_classes] | ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - task_mode (str): 任务模式，"CLS" 或 "MCQ"
            - pptLearner (PromptLearner): 提示学习器对象
        """
        if self._pptLearner is None:
            if pptLearner is not None:
                self.register_promptLearner(pptLearner) # 注册提示学习器
            else:
                raise ValueError("CoOp 的 提示学习器 pptLearner 为 None！")
        else:
            if pptLearner is not None:
                warnings.warn("CoOp 的 提示学习器 pptLearner 已经存在，此处将忽略传入的 pptLearner 对象！")
        
        suffixs = []  # 存储每个类别的后缀
        eos_indices = []  # 存储每个类别的后缀中的EOS的索引
        for cls in cls_list:
            suffix, eos_indice = self._pptLearner.construct_suffix(cls, clip_model=self) # suffix: Torch.tensor shape: (1, suffix_len, dim) ; eos_indice: int
            suffixs.append(suffix)
            eos_indices.append(eos_indice)
        suffixs = torch.cat(suffixs, dim=0)  # 包含每个类别的suffix | Torch.tensor shape: (n_cls, suffix_len, dim)
        self.eos_indices = torch.tensor(eos_indices, dtype=torch.long, device=self.device)  # 每个类别的 prompt 的结束符位置 | shape: (n_cls,)

        # 将 suffixs 和 eos_indices 赋值给 self.pptLearner
        if task_mode == "CLS":
            # 静态注册buffer（持久化保存），节省每次重复计算带来的开销
            self._pptLearner.register_buffer("suffixs", suffixs) # ' ' + 类别名 + '.' + [EOS] + '...' | shape: (n_cls, suffix_len, dim)
        elif task_mode == "MCQ":
            # 动态存储为实例属性（非持久化），每次前向传播时计算
            self._pptLearner.suffixs = suffixs # ' ' + 类别名 + '.' + [EOS] + '...' | shape: (n_cls, suffix_len, dim)
        else:
            raise ValueError(f"未知任务模式: {task_mode}")

    @property
    def pptLearner(self):
        """
        提示学习器属性
        """
        if self._pptLearner is None:
            raise ValueError("CoOp 的 提示学习器 pptLearner 为 None！")
        return self._pptLearner  


    def forward(self, image, return_feature=False):
        # --- 图像编码 ---
        image_features = self.image_encoder(image.type(self.dtype))  # [B, D]
        
        # --- 文本编码 ---
        B, num_choices = self._pptLearner.suffixs.shape[:2]
        
        # 扩展prefix和ctx到匹配批量维度
        prefixs = self._pptLearner.prefixs.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_cls, 1, D]
        ctx = self._pptLearner.ctx.unsqueeze(0).unsqueeze(1).expand(B, num_choices, -1, -1)  # [B, C, n_ctx, D]
        suffixs = self._pptLearner.suffixs  # [B, C, suffix_len, D]
        
        # 拼接prompts
        prompts = torch.cat([prefixs, ctx, suffixs], dim=2)  # [B, C, full_len=77, D] | 1 + n_ctx(4) + suffix_len(72)  = 77
        prompts = prompts.view(B * num_choices, *prompts.shape[2:])  # [B*C, full_len=77, D]

        # 位置编码
        t = prompts + self.positional_embedding.type(self.dtype)
        
        # 文本编码
        t = t.permute(1, 0, 2)  # [seq_len, B*C, D]
        t = self.text_encoder(t)
        t = t.permute(1, 0, 2)  # [B*C, seq_len, D]
        t = self.ln_final(t).type(self.dtype)
        
        # 提取EOS特征
        eos_indices = self.eos_indices.view(-1)  # [B*C]
        batch_indices = torch.arange(t.shape[0], device=self.device)
        EOS = t[batch_indices, eos_indices]  # [B*C, D]
        
        # 投影和归一化
        text_features = EOS @ self.text_projection
        text_features = text_features.view(B, num_choices, -1)  # [B, C, D]
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算相似度
        image_features = F.normalize(image_features, p=2, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features.unsqueeze(1) @ text_features.permute(0,2,1)  # [B, 1, D] @ [B, D, C] -> [B, C]
        
        return logits.squeeze(1)



    # def forward(self, image, return_feature=False):
    #     """ 
    #     重写 forward 函数 
        
    #     参数：
    #         - image(torch.Tensor): 输入图像，形状为 (batch_size, 3, height, width)
    #         - return_feature(bool): 是否返回图像特征，默认为 False

    #     主要步骤：
    #         1. 编码图像特征
    #         2. 生成每个类别的 prompt，拼接可训练的 prompts 和 可训练的位置嵌入 得到文本输入
    #         3. 编码文本特征
    #         4. 计算图像和文本之间的余弦相似度
    #         5. 返回与图像最相似的文本提示词对应的类别作为结果
    #     """
    #     # ---编码图像特征---
    #     image_features = self.image_encoder(image.type(self.dtype)) 
    #     # ---生成 prompt---
    #     prompts = self._pptLearner() # 获取学习到的每个类别的 prompt
    #     # ---编码文本特征---
    #     # 可训练的 prompts + 可训练的位置嵌入
    #     t = prompts + self.positional_embedding.type(self.dtype)
    #     # 交换维度，使其符合文本编码器输入格式 
    #     t = t.permute(1, 0, 2)  # (batch_size, n_ctx, width) -> (n_ctx, batch_size, width) | n_ctx(序列长度=num_patches+1) width(隐藏层宽度)
    #     t = self.text_encoder(t) # 编码文本特征
    #     t = t.permute(1, 0, 2)  # LND -> NLD
    #     t = self.ln_final(t).type(self.dtype) # (n_ctx, batch_size, width) -> (batch_size, n_ctx, width)
    #     # 获取文本编码特征
    #     batch_indices = torch.arange(t.shape[0])
    #     EOS = t[batch_indices, self.eos_indices]
    #     # 线性投影得到文本特征
    #     text_features = EOS @ self.text_projection
    #     # 归一化图像和文本特征 | dim=-1 表示在特征维度上进行归一化 p=2 表示使用 L2 范数进行归一化
    #     # print(f"图像特征：{image_features.shape}，文本特征：{text_features.shape}") # 图像特征：torch.Size([32, 512])，文本特征：torch.Size([100, 512])
    #     image_features = F.normalize(image_features, p=2, dim=-1)
    #     text_features = F.normalize(text_features, p=2, dim=-1)
    #     # ---计算 图像 和 文本 间的 余弦相似度---
    #     logit_scale = self.logit_scale.exp() # 温度参数 τ 的倒数
    #     logits_per_image = logit_scale * image_features @ text_features.t() # image->text 相似度 | [batch, num_classes]
    #     # ---返回结果---
    #     if return_feature: 
    #         return logits_per_image, image_features
    #     else:
    #         return logits_per_image
        

class PromptLearner(nn.Module):
    """ 
    提示学习器：用学习所有类别通用的上下文词ctx来生成每个类别的提示信息

    包含属性:
        - init_ctx_text(str): 初始的上下文词文本
        - n_ctx(int): 初始的上下文词文本的 words 数量
        - prompt的组成 (prefixs + ctx + suffixs) :
            - ctx: 可训练参数 上下文向量 ctx | torch.tensor | (n_ctx, dim)
            - prefixs: prompt的前缀: [SOS] | torch.tensor | (n_cls, 1, dim)
            - suffixs: prompt的后缀: " " + 类别名 + "." + [EOS] + '...'  | torch.tensor | (n_cls, *, dim)
    """
    def __init__(self, cfg, clip_model, n_cls):
        """
        初始化提示学习器

        参数：
            - cfg: 配置文件，包含模型的超参数和训练设置
            - classnames: 类别名称列表，用于生成提示信息 | 如 ['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
            - clip_model: 实例化的 CLIP 模型对象
            - n_cls: 类别数量
        
        配置：
        - cfg.MODEL.init_ctx: 初始化的上下文词，例如 "a photo of a"
        
        """
        super().__init__()
        # 将当前实例绑定到 CLIP 模型上
        self.device = clip_model.device  # CLIP 模型的设备
        self.dtype = clip_model.dtype  # CLIP 模型的数据类型
        clip_model.register_promptLearner(self)  # 将当前实例绑定到 CLIP 模型上
        
        # 初始化参数
        dtype = clip_model.dtype  # CLIP 模型的数据类型
        clip_imsize = clip_model.image_encoder.input_resolution # CLIP 模型的输入图像尺寸
        self.n_cls = n_cls  # 类别数量

        # 读取配置
        self.init_ctx_text = cfg.MODEL.init_ctx  # 预设的上下文词 (str) | 如 "a photo of a" | 所有类别通用的上下文词
        cfg_imsize = cfg.INPUT.SIZE # 配置文件中设定的输入图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) 必须等于 clip_imsize ({clip_imsize})" # 确保输入尺寸匹配
        
        # 将初始上下文词文本 init_ctx_text (str) 转为 可训练参数 上下文向量 ctx (Torch.tensor)
        init_ctx_text = self.init_ctx_text.replace("_", " ")  # 将下划线替换为空格
        self.n_ctx = len(init_ctx_text.split(" "))  # 上下文词包含的 token 数量 | 例如 "a photo of a" -> 4
        init_ctx_token = _tokenizer.encode(init_ctx_text) # token 列表 | list<int>
        init_ctx_token_tensor = torch.tensor(init_ctx_token, dtype=torch.long).unsqueeze(0).to(clip_model.device)  # shape: (1, n_ctx)
        with torch.no_grad():
            ctx_vectors = clip_model.token_embedding(init_ctx_token_tensor).type(dtype)  # shape: (1, n_ctx, dim)
        ctx_vectors = ctx_vectors.squeeze(0)  # 去掉 batch 维度，变为 shape: (n_ctx, dim)
        self.ctx = nn.Parameter(ctx_vectors) # (Torch.tensor) shape: (n_ctx, dim)

        # prompt 前缀 self.prefixs -> <SOS> 的 嵌入表示向量 sot_embedding (Torch.tensor)
        sos_token = _tokenizer.encoder["<|startoftext|>"]  # token (int)
        sos_tensor = torch.tensor([sos_token], dtype=torch.long).to(clip_model.device)  # shape: (1,)
        with torch.no_grad():
            SOS = clip_model.token_embedding(sos_tensor).type(dtype)  # (Torch.tensor) shape: (1, dim)
        prefixs_buffer = SOS.unsqueeze(0).expand(self.n_cls, -1, -1).clone()  # [SOS] | torch.tensor | shape: (n_cls, 1, dim)
        self.register_buffer('prefixs', prefixs_buffer)  # 注册 prefixs 为缓冲区，表示固定的、不参与训练的张量，节省显存
        
        # prompt 后缀 suffixs | 待外部调用 construct_suffix 进行 赋值 或 register
        # self.suffixs = None  # 类别名 + [EOS] token | torch.tensor | shape: (n_cls, *, dim) | 直接使用register_buffer 进行注册


    # ==================批量生成后缀==================

    def batch_construct_suffix(self, class_list, clip_model):
        """
        批量生成后缀
        """
        # 计算长度约束（与construct_suffix保持一致）
        context_length = 77  # CLIP固定上下文长度
        sot_length = 1       # [SOS] token
        target_length = context_length - sot_length - self.n_ctx  # suffix部分最大允许长度
        max_cls_length = target_length - 1  # 为EOS保留一个位置

        # 批量处理所有类别
        tokens_list = []
        eos_indices = []
        
        for cls in class_list:
            # 预处理类别名称，替换下划线为空格，并添加EOS
            cls_text = cls.replace("_", " ")
            cls_tokens = _tokenizer.encode(cls_text)[:max_cls_length]
            tokens = cls_tokens + [_tokenizer.encoder["<|endoftext|>"]]
            
            # 计算填充长度
            pad_length = target_length - len(tokens)
            tokens += [0] * pad_length
            
            # 记录EOS位置（相对于整个prompt）
            eos_in_suffix = len(cls_tokens)
            full_eos_pos = 1 + self.n_ctx + eos_in_suffix  # SOT(1) + ctx长度 + 在suffix中的位置
            eos_indices.append(full_eos_pos)
            
            tokens_list.append(tokens)

        # 转换为张量
        token_tensor = torch.tensor(tokens_list, dtype=torch.long, device=self.device)  # [N, target_length]
        
        # 批量嵌入
        with torch.no_grad():
            embeddings = clip_model.token_embedding(token_tensor).type(self.dtype)  # [N, target_length, D]
        
        eos_indices = torch.tensor(eos_indices, dtype=torch.long, device=self.device)
        
        return embeddings, eos_indices # 后缀：torch.Size([128, 72, 512])，EOS索引：torch.Size([128])


    # ===============================================
    

    # def construct_suffix(self, cls, clip_model):
    #     """
    #     (由外部调用)
    #     构造 prompt 中 关于cls类别的 后缀部分 suffix | Torch.tensor (*, dim)
    #     - 完整的 prompt 组成为：[SOS] + 上下文ctx + " " + 类别名 + "." + [EOS] + '...'
    #         - suffix 为 < 类别名 + [EOS] + '...' >
    #             - '...' 为0填充，以保证 prompt 的 长度统一为 context_length (Clip默认为77)
        
    #     参数：
    #         - cls (str): 类别名称，例如 "sleeping_dog"

    #     返回：
    #         - suffix (Torch.tensor): prompt 中 类别相关的 后缀部分 | Torch.tensor | shape: (1, *, dim)
    #         - eos_indice (int): 每个类别的 prompt 的结束符 [EOS] 在 完整prompt 中的 位置索引 | int
    #     """
    #     cls_text = cls.replace("_", " ")  # (str) 预处理类别名称，替换下划线为空格，并添加句号
    #     # 使用全局的_tokenizer进行分词
    #     eos_token = _tokenizer.encoder["<|endoftext|>"]
    #     cls_tokens = _tokenizer.encode(cls_text)
        
    #     # 计算目标长度
    #     target_length = 77 - 1 - self.n_ctx  # 77 - SOT(1) - ctx长度(n_ctx)
    #     max_cls_length = target_length - 1   # 为EOS保留一个位置
        
    #     # 截断类别名token以避免溢出
    #     cls_tokens = cls_tokens[:max_cls_length]
        
    #     # 组合token并添加填充
    #     tokens = cls_tokens + [eos_token]
    #     pad_length = target_length - len(tokens)
    #     tokens += [0] * pad_length  # 用0填充剩余位置
        
    #     # 转换为张量并移至对应设备
    #     token_tensor = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
    #     # 获取嵌入向量
    #     with torch.no_grad():
    #         embeddings = clip_model.token_embedding(token_tensor).type(self.dtype)
        
    #     # 计算EOS在prompt中的位置
    #     eos_in_suffix = len(cls_tokens)
    #     eos_indice = 1 + self.n_ctx + eos_in_suffix  # 1(SOT) + n_ctx(ctx长度) + 在suffix中的位置
        
    #     # 调整形状以匹配后续拼接
    #     suffix = embeddings.unsqueeze(0)  # shape: (1, *, dim)

    #     return suffix, eos_indice  # 所有类别的 suffix 构成 self.suffixs

    def forward(self):
        # 构造promt | 类别名称放在结尾（论文实验显示"end"效果最好）
        prompts = torch.cat(
            [
                self.prefixs,  # (n_cls, 1, dim) [SOS] | shape: (n_cls, 1, dim)
                self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1),  # (n_cls, n_ctx, dim) 上下文 | shape: (n_cls, n_ctx, dim)
                self.suffixs,  # (n_cls, *, dim) 包括 类别名+[EOS] | shape: (n_cls, *, dim)
            ],
            dim=1,
        )
        return prompts  # [SOS] + 可学习上下文 + (" "+类别名+".") + [EOS] | Torch.tensor | shape: (n_cls, *, dim)