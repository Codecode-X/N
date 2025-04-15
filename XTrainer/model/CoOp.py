from .Clip import Clip, tokenize
from utils import SimpleTokenizer as Tokenizer
from torch import nn
import torch.nn.functional as F
import torch
from .build import MODEL_REGISTRY
from .ModelBase import ModelBase

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

    
    def batch_init_prompt_learner(self, batch_choices):
        """
        用于MCQ任务，批量初始化提示学习器
        - 通过批量类别名称列表生成提示学习器的后缀部分，并将其存储在提示学习器中
        - 在每次forward()时，由外部trainer调用，使用当前批次的类别名称生成当前批次的prompt的后缀部分

        参数：
            - batch_choices (list): 批量类别名称列表，形状为 [batch_size, num_choices]
        
        返回：
            - None
        """
        # 展平所有选项
        all_classes = [cls for choices in batch_choices for cls in choices]  # shape:(B*C=128, ) | 128 = 4(num_choices) * 32(batchsize)
        
        # 批量生成后缀 | suffixs:(batchsize*n_cls, target_length=72, dim); eos_indices:(batchsize*n_cls, )
        suffixs, eos_indices = self._pptLearner.batch_construct_suffix(all_classes, self)
        
        # 展开为 suffixs: [batch_size, n_cls, target_length, dim] 和 eos_indices: [batch_size, n_cls]
        num_choices = len(batch_choices[0])
        self._pptLearner.suffixs = suffixs.view(len(batch_choices), num_choices, *suffixs.shape[1:]) # [batch_size, num_choices, target_length=72, dim]
        self.eos_indices = eos_indices.view(len(batch_choices), num_choices) # [batch_size, num_choices]

    def init_prompt_learner(self, cls_list):
        """
        用于CLS任务，初始化提示学习器
        - 通过类别名称列表生成提示学习器的后缀部分，并将其存储在buffer中
        - 在初始化时，由外部trainer调用，使用所有类别名称生成所有类别的prompt的后缀部分
        
        参数：
            - cls_list (list): 类别名称列表，形状为 [n_cls]，例如['pagoda', 'panda', ..., 'stegosaurus', 'stop_sign']
        
        返回：
            - None

        得到：
            - suffixs (tensor): 后缀的嵌入表示 | shape: (1, n_cls, target_length=72, dim)
            - eos_indices (tensor): EOS token 在完整提示中的索引 | shape: (1, n_cls, )
        """
        # 生成后缀 | suffixs:(n_cls, target_length=72, dim); eos_indices:(n_cls, )
        suffixs, eos_indices = self._pptLearner.batch_construct_suffix(cls_list, self)
        
        # 增加 batch 维度
        suffixs = suffixs.unsqueeze(0)  # [1, n_cls, target_length=72, dim]
        eos_indices = eos_indices.unsqueeze(0)  # [1, n_cls]
        
        # 注册后缀和EOS索引
        self._pptLearner.register_buffer('suffixs', suffixs) # [1, n_cls, target_length=72, dim]
        self.register_buffer('eos_indices', eos_indices) # [1, n_cls]

    @property
    def pptLearner(self):
        """
        提示学习器属性
        """
        if self._pptLearner is None:
            raise ValueError("CoOp 的 提示学习器 pptLearner 为 None！")
        return self._pptLearner  


    def forward(self, image):
        # --- 图像编码 ---
        image_features = self.image_encoder(image.type(self.dtype))  # [B, D]
        
        # --- 文本编码 ---
        suffixs = self._pptLearner.suffixs  # [B, C, suffix_len, D]
        B, num_choices = suffixs.shape[:2]
        
        # 扩展prefix和ctx到匹配批量维度
        prefixs = self._pptLearner.prefixs.unsqueeze(0).expand(B, -1, -1, -1)  # [B, C, 1, D]
        ctx = self._pptLearner.ctx.unsqueeze(0).unsqueeze(1).expand(B, num_choices, -1, -1)  # [B, C, n_ctx, D]

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
        logits = logit_scale * image_features.unsqueeze(1) @ text_features.permute(0,2,1)  # [B, 1, D] @ [B, D, C] -> [B, 1, C]
        
        return logits.squeeze(1) # [B, 1, C] -> [B, C]
        

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
        # self.suffixs = None  # 类别名 + [EOS] token | torch.tensor | shape: CLS(n_cls, *, dim) or MCQ((B*n_cls), *, dim)| 直接使用register_buffer 进行注册


    def batch_construct_suffix(self, class_list, clip_model):
        """
        批量生成后缀

        参数：
            - class_list (list): 类别名称列表，例如 ["sleeping_dog", "running_cat", ...]
            - clip_model: 实例化的 CLIP 模型对象

        返回：
            - suffix_embeddings (torch.Tensor): 后缀的嵌入表示 | shape: (batchsize*n_cls, target_length, dim) | 其中 target_length = 77 - 1 - n_ctx
            - eos_indices (torch.Tensor): EOS token 在完整提示中的索引 | shape: (batchsize*n_cls, )
        
        主要步骤：
            1. 计算后缀的长度约束
            2. 批量处理所有类别，生成后缀的嵌入表示
            3. 返回后缀的嵌入表示和 EOS token 的索引
        """
        # 计算长度约束（与construct_suffix保持一致）
        context_length = 77  # CLIP固定上下文长度
        sos_length = 1       # [SOS] token
        target_length = context_length - sos_length - self.n_ctx  # suffix部分最大允许长度
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
            suffix_embeddings = clip_model.token_embedding(token_tensor).type(self.dtype)  # [N, target_length, D]
        
        eos_indices = torch.tensor(eos_indices, dtype=torch.long, device=self.device)
        
        return suffix_embeddings, eos_indices # 后缀：torch.Size([128, 72, 512])，EOS索引：torch.Size([128])


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