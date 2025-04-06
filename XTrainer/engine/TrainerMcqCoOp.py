from .base_class.TrainerMcqBase import TrainerMcqBase
from model import build_model
from utils import count_num_param, load_checkpoint
from torch.cuda.amp import GradScaler
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.metrics import compute_accuracy
from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
from .build import TRAINER_REGISTRY
import os.path as osp
from model.CoOp import PromptLearner
import tqdm


@TRAINER_REGISTRY.register()
class TrainerMcqCoOp(TrainerMcqBase):
    """CoOp 处理 MCQ 任务的 Trainer 类。"""

    def init_model(self, cfg):
        """
        (实现父类的方法) 初始化模型。
        -> 只训练 Prompt生成器 示例

        参数：
            cfg (CfgNode): 配置。

        类包含的属性：
            - CoOp_model (nn.Module): CoOp 模型。
            - pptLearner (PromptLearner): 提示学习器。
            - scaler (GradScaler): 自动混合精度训练的缩放器。(可选)
            - optim (Optimizer): 优化器。
            - sched (LRScheduler): 学习率调度器。

        返回：
            - model (nn.Module): 模型。
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

        # 初始化提示学习器，并注册到CoOp_model中，用于学习提示信息
        self.task_type = cfg.TASK_TYPE # "CLS"分类；"MCQ"多选
        self.pptLearner = PromptLearner(cfg, self.CoOp_model)

        # 将模型调整为精度混合训练，以减少显存占用 (如果配置了精度混合训练)
        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # 多 GPU 并行训练情况，则将模型部署到多个 GPU 上 (如果有多个 GPU)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.CoOp_model = nn.DataParallel(self.CoOp_model)

        # 构建 PromptLearner 并注册 -> 示例：优化器只优化 CLIP 的 PromptLearner
        promptLearner = self.CoOp_model.pptLearner
        self.optim = build_optimizer(promptLearner, cfg)
        self.sched = build_lr_scheduler(cfg, self.optim)
        self.register_model("CLIP_promptLearner", promptLearner, self.optim, self.sched)

        # 给 promptLearner 载入 提示学习器 的预训练权重 (如果配置了 预训练权重) - 请仅使用相同数据集的预训练权重，否则没有意义
        if cfg.MODEL.INIT_WEIGHTS_PATH: 
            pretarined_path = cfg.MODEL.INIT_WEIGHTS_PATH
            print(f"载入预训练权重：{pretarined_path}")
            self.load_model(directory=pretarined_path)  # 加载最佳模型

        return self.CoOp_model, self.optim, self.sched
    
    def forward_backward(self, batch): 
        """
        (实现父类的方法) 前向传播和反向传播。
        """
        image, num_choices, choices, correct_answer, correct_answer_type = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
        assert image is not None and num_choices > 1, "forward_backward() 中 parse_batch_train 解析到的图像为空或者num_choices<=1"

        self.CoOp_model.init_promptLearner(cls_list=choices, task_mode=self.task_type) # 根据choices初始化提示学习器的文本prompt
        label = torch.tensor(correct_answer, dtype=torch.long, device=self.device) # [batch] -> 正确答案索引
        
        # 图像-image: [batch, 3, 224, 224]
        logits = self.CoOp_model(image) # [batch, num_classes] -> 模型预测的各个类别的置信度
        loss = F.cross_entropy(logits, label)  # 计算损失  
        self.model_backward_and_update(loss)  # 反向传播

        # 需要记录的 loss 日志
        loss_summary = {  
            "loss": loss.item(),
            "acc": compute_accuracy(logits, label)[0].item(),
        }

        # 到阶段自动更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    @torch.no_grad()
    def test(self, split=None):
        """
        测试 (需要子基类根据任务特点重写)。 

        主要步骤：
            1. 设置模型模式为 eval，重置评估器。
            2. 确定测试集（val or test，默认为测试集）。
            3. 初始化统计计数器。
            4. 开始测试。
            5. 模型推理结果分析。
                - 计算总体准确率。
                - 计算每种类型的准确率。
                - 找出最常见的错误答案类型。
                - 计算每种错误答案类型的百分比。
            6. 返回包含所有计算指标的字典。
        """

        # --------设置模型模式为 eval, 重置评估器--------
        self.set_model_mode("eval")
        self.evaluator.reset() # 重置评估器

        # --------确定测试集（val or test，默认为测试集）--------
        if split is None: # 如果 split 为 None，则使用配置中的测试集
            split = self.cfg.TEST.SPLIT 
        if split == "val" and self.val_loader is not None: 
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"在 *{split}* 集上测试")

        # --------开始测试--------
        for batch_idx, batch in enumerate(tqdm(data_loader)): # 遍历数据加载器
            image, num_choices, choices, correct_answer, correct_answer_type = self.parse_batch_train(batch)  # 解析训练批次数据，获取图像和标签
            assert image is not None and num_choices > 1, "forward_backward() 中 parse_batch_train 解析到的图像为空或者num_choices<=1"
            
            self.CoOp_model.init_promptLearner(cls_list=choices, task_mode=self.task_type) # 根据choices初始化提示学习器的文本prompt            
            # 图像-image: [batch, 3, 224, 224]
            logits = self.CoOp_model(image) # [batch, num_classes] -> 模型预测的各个类别的置信度
            
            # 获取预测的答案索引，并进行评估
            logits = torch.argmax(logits, dim=1)
            labels = torch.tensor(correct_answer, dtype=torch.long, device=self.device) # [batch] -> 正确答案索引
            self.evaluator.process(logits, labels, correct_answer, correct_answer_type) # 评估器评估模型输出和标签
            
        # --------模型推理结果分析--------
        results = self.evaluator.evaluate() # 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        for k, v in results.items(): 
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch) # 记录到 tensorboard

        return list(results.values())[0] # 返回第一个值：accuracy

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
            state_dict['prefixs'] = state_dict['token_prefix']
            state_dict['suffixs'] = state_dict['token_suffix']

            try:
                self._models[name].load_state_dict(state_dict) # 加载模型状态字典
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print(f"Warning: {e}. 预训练模型和当前模型的num_classes不匹配，请不要使用其他数据集的预训练权重！")
                else:
                    raise e
        
        return epoch # 返回训练轮数
            
