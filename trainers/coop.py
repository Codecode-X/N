import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# 加载 CLIP 到 cpu
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME  # 从配置中获取骨干网络名称
    url = clip._MODELS[backbone_name]  # 获取该模型对应的下载地址
    model_path = clip._download(url)  # 下载模型并返回本地路径
    try:
        # 尝试以 TorchScript JIT 方式加载模型
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None  # 如果 JIT 加载成功，则不需要 state_dict
    except RuntimeError:
        # 如果 JIT 加载失败，则改用普通方式加载 state_dict
        state_dict = torch.load(model_path, map_location="cpu")
    # 构建 CLIP 模型
    model = clip.build_model(state_dict or model.state_dict())
    return model


# 文本编码器 Encode (可训练的 prompts + 可训练的位置嵌入)->EOT 特征表示全局特征
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # 从 CLIP 预训练模型中提取文本编码相关的模块
        self.transformer = clip_model.transformer  # Transformer 编码器
        self.positional_embedding = clip_model.positional_embedding # 位置嵌入
        self.ln_final = clip_model.ln_final  # LN 层
        self.text_projection = clip_model.text_projection  # 线性投影层
        self.dtype = clip_model.dtype  # 数据类型（float16 / float32）

    def forward(self, prompts, eot_indices):
        """
            prompts: 可学习的每个类别对应的提示词
            eot_indices: prompt 结束符位置
        """
        # 可训练的 prompts + 可训练的位置嵌入
        x = prompts + self.positional_embedding.type(self.dtype)
        
        # 交换维度，使其符合 Transformer 输入格式 
        x = x.permute(1, 0, 2)  # (batch_size, n_ctx, width) -> (n_ctx, batch_size, width) | n_ctx(序列长度=num_patches+1) width(隐藏层宽度)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype) # (n_ctx, batch_size, width) -> (batch_size, n_ctx, width)

        # 获取文本编码特征
        batch_indices = torch.arange(x.shape[0])
        EOT = x[batch_indices, eot_indices]

        # 线性投影
        x = EOT @ self.text_projection

        return x


# 可学习 Prompt
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        # 读取配置
        n_cls = len(classnames)  # 类别数量
        n_ctx = cfg.TRAINER.COOP.N_CTX  # 上下文词数
        ctx_init = cfg.TRAINER.COOP.CTX_INIT  # 是否使用预设的上下文词
        dtype = clip_model.dtype  # CLIP 模型的数据类型
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 获取上下文词向量的维度
        clip_imsize = clip_model.visual.input_resolution # CLIP 模型的输入图像尺寸
        cfg_imsize = cfg.INPUT.SIZE[0] # 配置文件中设定的输入图像尺寸
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})" # 确保输入尺寸匹配
        
        """初始化 上下文前缀词 以及其 嵌入向量"""
        # 如果提供了初始上下文前缀词 ctx_init，则初始化上下文前缀词为 ctx_init，上下文前缀向量为 clip.tokenize(ctx_init)
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")  # 将下划线替换为空格
            n_ctx = len(ctx_init.split(" "))  # 重新计算上下文词数量
            tokenized_ctx = clip.tokenize(ctx_init)  # 将文字形式的 ctx_init 分词编码为 token
            with torch.no_grad():
                ctx_embedding = clip_model.token_embedding(tokenized_ctx).type(dtype)  # 获取嵌入表示 (batch_size, seq_len, embedding_dim)
            ctx_vectors = ctx_embedding[0, 1:1+n_ctx, :]  # 提取上下文嵌入向量 | 1:1+n_ctx: 索引 0 位置对应 [SOS]
            prompt_prefix = ctx_init  # 设定上下文前缀就是初始上下文词 ctx_init
       
        # 如果没有提供初始上下文前缀词，则随机初始化上下文前缀词为 XXXXXX，上下文前缀向量为正态分布随机数
        else:
            # 随机初始化
            if cfg.TRAINER.COOP.CSC: # 每个类别初始化特定的上下文
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype) # (num_class, seq_len, embedding_dim)
            else: # 所有类别共享一个上下文
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  # (seq_len, embedding_dim)
            nn.init.normal_(ctx_vectors, std=0.02)  # 进行正态分布初始化
            prompt_prefix = " ".join(["X"] * n_ctx)  # 设定占位的上下文前缀词 XXXXXX

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        """构造 每个类别 完整的 prompt 文本（前缀词 + 类别名 + 句号）"""
        classnames = [name.replace("_", " ") for name in classnames]  # 处理类别名称中的"_"
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]  # 计算每个类别名称的 token 数量
        prompts = [prompt_prefix + " " + name + "." for name in classnames]  # 生成完整的 prompt 文本（前缀词 + name + 句号）
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # 将文字形式的 prompt 分词编码为 token
        eot_indices = tokenized_prompts.argmax(dim=-1)  # 获取 tokenized_prompts 的 [EOS] token 的 索引
        with torch.no_grad():
            ctx_embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # 得到 tokenized_prompts 的嵌入表示

        # 为了在训练和推理时保留 学习到的上下文信息，并确保它们在模型的存储和加载过程中保持一致性。
        # 如果没有这些信息，每次加载模型时，模型就会丢失与类别相关的上下文，从而影响推理结果的准确性。
        self.register_buffer("token_prefix", ctx_embedding[:, :1, :])  # [SOS]
        self.register_buffer("token_suffix", ctx_embedding[:, 1 + n_ctx :, :])  # 包括 类别名 和 [EOS]

        self.n_cls = n_cls  # 类别数
        self.n_ctx = n_ctx  # 上下文词数
        self.name_lens = name_lens  # 记录每个类别名称的长度 (token 数)
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION  # 记录类别名称插入的位置（前、中间、后）
        
        # 将上下文向量 ctx（默认是可训练前缀）设为 可训练参数
        self.learnable_ctx = nn.Parameter(ctx_vectors)

        # 记录每一类的 prompt 的结尾 EOT 索引位置
        self.eot_indices = eot_indices

    def forward(self):
        learnable_ctx = self.learnable_ctx  # 取出上下文向量
        if learnable_ctx.dim() == 2: # 只存在于所有类别都用同样的前缀
            learnable_ctx = learnable_ctx.unsqueeze(0).expand(self.n_cls, -1, -1)  # 维度适配

        # register_buffer 保存的 prefix 和 suffix
        prefix = self.token_prefix  # [SOS]
        suffix = self.token_suffix  # 包括 类别名 和 [EOS]
        
        # 类别名称放在结尾（论文实验显示"end"效果最好）
        if self.class_token_position == "end": 
            prompts = torch.cat(  # [SOS] + 上下文 + 类别名 + [EOS]
                [
                    prefix,  # (n_cls, 1, dim) [SOS]
                    learnable_ctx,     # (n_cls, n_ctx, dim) 学习到的上下文
                    suffix,  # (n_cls, *, dim) 包括 类别名+[EOS]
                ],
                dim=1,
            )
        else: raise NotImplemented('论文实验显示"end"效果最好，因此其他未实现')
        return prompts # 完整的 prompts -> [SOS] + 上下文 + 类别名 + [EOS]


# 自定义的 CLIP 模型类
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # 提示学习
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model) # 初始化一个 PromptLearner 对象，用于学习提示信息
        self.eot_indices = self.prompt_learner.eot_indices  # 每个类别的 prompt 的结束符位置 
        # 常规 clip
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        # 编码图像特征
        image_features = self.image_encoder(image.type(self.dtype))  
        # 编码文本特征
        prompts = self.prompt_learner() # 获取学习到的 每个类别的 prompt
        text_features = self.text_encoder(prompts, self.eot_indices)
        # 计算相似度
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits # 相似度


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    
    def check_cfg(self, cfg): # 检查配置文件中的 PREC 字段是否为合法值
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    
    # 构建模型
    def build_model(self):
        cfg = self.cfg  # 配置
        classnames = self.dm.dataset.classnames # 类别名称

        # 加载模型
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg) # 导入预训练的 clip 
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float() # CLIP's default precision is fp16
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # 冻结 图像编码器 和 文本编码器
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # 如果 prompt_learner 有预训练权重就导入
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # 优化器只优化 prompt_learner
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        
        # GradScaler 是 PyTorch 里用于自动混合精度训练的工具，它能够在不损失模型精度的前提下，减少训练时的内存占用与计算量。
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # 多 GPU 并行训练情况
        # Note that multi-gpu training could be slow because CLIP's size is big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    # 前向和反向传播的过程    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
        
        prec = self.cfg.TRAINER.COOP.PREC  # 配置的精度
        if prec == "amp":  # 自动混合精度训练
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:  # 默认 fp16
            output = self.model(image) # 模型预测
            loss = F.cross_entropy(output, label)  # 计算损失
            self.model_backward_and_update(loss)  # 反向传播

        # 需要记录的 loss 日志
        loss_summary = {  
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # 到阶段自动更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    # 解析一个 batch 的数据，返回 输入 和 标签
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    # 载入预训练模型权重
    def load_model(self, directory, epoch=None):
        # 如果没有给出预训练模型的 directory，则直接返回
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
                
        # 默认加载最佳模型权重
        model_file = "model-best.pth.tar"
        if epoch is not None:  # 加载指定 epoch 的模型权重
            model_file = "model.pth.tar-" + str(epoch)
        
        names = self.get_model_names()
        for name in names:
            model_path = osp.join(directory, name, model_file) # 生成模型文件的完整路径
            if not osp.exists(model_path): # 检查模型文件是否存在
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint = load_checkpoint(model_path) # 加载检查点 - 包含模型的 状态字典 和训练的 epoch 信息
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # 加载模型权重
            # 但不加载 token_prefix，token_suffix，只加载可学习的上下文向量 ctx
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]  # SOS
            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]  # 类别名+EOS
            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
