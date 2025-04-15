from .TrainerClsClip import TrainerClsClip
from model import build_model
from utils import count_num_param, load_checkpoint
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.metrics import compute_accuracy
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from .build import TRAINER_REGISTRY
import os.path as osp
from model.CoOp import PromptLearner


@TRAINER_REGISTRY.register()
class TrainerClsCoOp(TrainerClsClip):
    """CoOp 处理分类任务的 Trainer 类。"""
    def init_model(self, cfg):
        """
        (实现父类的方法) 初始化模型。
        -> 只训练 Prompt生成器 示例

        参数：
            cfg (CfgNode): 配置。

        返回：
            CoOp_model (nn.Module): CoOp 模型。
            optim (Optimizer): 优化器。
            sched (LRScheduler): 学习率调度器。

        类包含的属性：
            - CoOp_model (nn.Module): CoOp 模型。
            - pptLearner (PromptLearner): 提示学习器。
            - optim (Optimizer): 优化器。
            - sched (LRScheduler): 学习率调度器。

        主要步骤：
        1. 构建模型
        2. 冻结 CLIP 的文本编码器和图像编码器，只训练 CoOp 的 PromptLearner
        3. 将模型移动到设备
        4. 将模型调整为精度混合训练 (如果配置了精度混合训练)
        5. 多 GPU 并行训练情况，则将模型部署到多个 GPU 上 (如果有多个 GPU)
        6. 构建优化器和调度器，只优化 CLIP 的 PromptLearner，并注册
        7. 返回模型、优化器和调度器
        """
        # 构建模型
        assert cfg.MODEL.NAME == "CoOp", f"TrainerClsCoOp 只支持 CoOp 模型，但 cfg.MODEL.NAME = {cfg.MODEL.NAME}"
        self.CoOp_model = build_model(cfg) # 构建模型 (此处 CLIP 模型提供了预训练模型的载入)
        print("模型参数数量：", count_num_param(self.CoOp_model))

        # 冻结模型某些层 -> 示例：冻结 CLIP 的文本编码器和图像编码器，只训练 CoOp 的 PromptLearner
        if cfg.TRAINER.FROZEN:
            for name, param in self.CoOp_model.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

        # 将模型移动到设备
        self.CoOp_model.to(self.device)

        # 设置模型的文本标签，初始化提示学习器
        sorted_labels = sorted(self.dm.lab2cname.items(), key=lambda x: x[0]) # 将文本标签按照 label 从小到大排序，方便模型预测结果与 label 进行对齐
        label_texts = [item[1] for item in sorted_labels]  # 文本标签 tensor | [num_classes]
        print("从小到大排序后的数据集文本标签：", label_texts)
        self.pptLearner = PromptLearner(cfg, self.CoOp_model, n_cls=len(label_texts))  # 初始化一个 PromptLearner 对象，并注册到CoOp_model中，用于学习提示信息
        self.CoOp_model.init_prompt_learner(cls_list=label_texts) # 初始化提示学习器

        # 多 GPU 并行训练情况，则将模型部署到多个 GPU 上 (如果有多个 GPU)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.CoOp_model = nn.DataParallel(self.CoOp_model)

        # 构建 PromptLearner 并注册 -> 示例：优化器只优化 CLIP 的 PromptLearner
        self.optim = build_optimizer(self.pptLearner, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_promptLearner", self.pptLearner, self.optim, self.sched)

        # 给 promptLearner 载入 提示学习器 的预训练权重 (如果配置了 预训练权重) - 请仅使用相同数据集的预训练权重，否则没有意义
        if hasattr(cfg.MODEL, "INIT_WEIGHTS_PATH") and cfg.MODEL.INIT_WEIGHTS_PATH: 
            pretarined_path = cfg.MODEL.INIT_WEIGHTS_PATH
            print(f"载入预训练权重：{pretarined_path}")
            self.load_model(directory=pretarined_path)  # 加载最佳模型

        return self.CoOp_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (实现父类的方法) 前向传播和反向传播。
        """
        image, label = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
        assert image is not None and label is not None, "forward_backward() 中 parse_batch_train 解析到的图像和标签不能为空"

        prec = self.cfg.TRAINER.PREC  # 配置的精度 # 默认 fp16
        if prec != "fp16": Warning.warn(f"TrainerClsCoOp 只支持 fp16 精度，但 cfg.TRAINER.PREC = {prec}，此处仍使用 fp16 精度以匹配Clip")
        
        output = self.CoOp_model(image) # 模型预测
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


    def load_model(self, directory, epoch=None):
        """
        重写父类方法-工具方法：将 dictionary 中的模型文件载入到模型中。
        
        解决权重文件中预训练模型的num_classes和当前模型的num_classes不匹配的问题: 裁剪 token_prefix 和 token_suffix。
        
        参数：
            * directory: 模型目录
            * epoch: epoch | 为None 时，加载最佳模型
        
        返回：
            * epoch: 训练轮数

        加载内容：
            * 模型状态字典
            * epoch
            * 验证结果
        """
        assert directory is not None, "load_model()的参数directory模型目录为None"

        names = self.get_model_names()  # 获取所有模型名称

        model_file = "model-best.pth.tar" # 默认情况下，加载最佳模型
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch) # 如果指定 epoch，加载指定 epoch 的模型

        # 遍历所有模型名称，加载模型
        for name in names: 
            model_path = osp.join(directory, name, model_file) # 模型路径

            if not osp.exists(model_path): # 如果模型路径不存在，抛出异常
                raise FileNotFoundError(f"没有找到模型文件 在 {model_path}!")

            checkpoint = load_checkpoint(model_path) # 加载检查点
            state_dict = checkpoint["state_dict"] # 获取状态字典
            epoch = checkpoint["epoch"] # 获取 epoch

            try:
                self._models[name].load_state_dict(state_dict) # 加载模型状态字典
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Warning: {e}. 预训练模型的num_classes和当前模型的num_classes不匹配，尝试裁剪预训练模型的token_prefix和token_suffix")
                    # 裁剪 token_prefix 和 token_suffix
                    state_dict["token_prefix"] = state_dict["token_prefix"][:self._models[name].token_prefix.shape[0]]  # 裁剪成 [1000, 1, 512]->[num_classes, 1, 512] 都是一样的前缀和后缀
                    state_dict["token_suffix"] = state_dict["token_suffix"][:self._models[name].token_prefix.shape[0]]  # 裁剪成 [1000, 1, 512]->[num_classes, 72, 512]
                    self._models[name].load_state_dict(state_dict) # 加载模型状态字典
                else:
                    raise e
        
        return epoch # 返回训练轮数
            
