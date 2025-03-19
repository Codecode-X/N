import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator


class SimpleNet(nn.Module):
    """一个简单的神经网络，由一个 CNN 骨干网络和一个可选的头部（如用于分类的 MLP）组成。"""

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        # 构建骨干网络
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME, # 骨干网络的名字
            verbose=cfg.VERBOSE, # 是否打印信息
            pretrained=model_cfg.BACKBONE.PRETRAINED, # 是否加载预训练模型
            **kwargs, # 其他参数
        )
        fdim = self.backbone.out_features # 骨干网络的输出特征维度

        self.head = None
        # 如果配置了头部网络，则构建头部网络
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME, # 头部网络的名字
                verbose=cfg.VERBOSE, # 是否打印信息
                in_features=fdim, # 输入特征维度
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS, # 隐藏层
                activation=model_cfg.HEAD.ACTIVATION, # 激活函数
                bn=model_cfg.HEAD.BN, # 是否使用 BatchNorm
                dropout=model_cfg.HEAD.DROPOUT, # Dropout 概率
                **kwargs, # 其他参数
            )
            fdim = self.head.out_features # 头部网络的输出特

        self.classifier = None 
        # 如果类别数大于 0，则构建分类器
        if num_classes > 0: # 如果类别数大于 0
            self.classifier = nn.Linear(fdim, num_classes) # 构建分类器

        self._fdim = fdim # 头部网络的输出特征维度

    @property
    def fdim(self):
        """返回头部网络的输出特征维度"""
        return self._fdim

    def forward(self, x, return_feature=False):
        # 前向传播，通过骨干网络
        f = self.backbone(x)
        # 如果有头部网络，通过头部网络
        if self.head is not None:
            f = self.head(f)

        # 如果没有分类器，返回特征
        if self.classifier is None:
            return f

        # 通过分类器得到输出
        y = self.classifier(f)

        # 如果需要返回特征，返回输出和特征
        if return_feature:
            return y, f

        # 返回输出
        return y


class TrainerBase:
    """迭代训练器的基类。"""

    def __init__(self):
        self._models = OrderedDict()  # 存储模型的有序字典
        self._optims = OrderedDict()  # 存储优化器的有序字典
        self._scheds = OrderedDict()  # 存储学习率调度器的有序字典
        self._writer = None  # TensorBoard 的 SummaryWriter

    def register_model(self, name="model", model=None, optim=None, sched=None):
        """工具方法：注册模型、优化器和学习率调度器。"""
        
        # 确保在调用 super().__init__() 之后才能注册模型
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )
        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )
        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )
        assert name not in self._models, "Found duplicate model names"  # 确保模型名称不重复

        # 注册
        self._models[name] = model  # 注册模型
        self._optims[name] = optim  # 注册优化器
        self._scheds[name] = sched  # 注册学习率调度器

    def get_model_names(self, names=None):
        """工具方法：获取所有模型名称。"""
        names_real = list(self._models.keys())  # 获取所有模型名称
        if names is not None:
            names = tolist_if_not(names)  # 如果 names 不是列表，将其转换为列表
            for name in names:
                assert name in names_real  # 确保名称在已注册的模型名称中
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        """工具方法：保存模型。"""
        names = self.get_model_names()  # 获取所有模型名称
        for name in names:
            model_dict = self._models[name].state_dict()  # 获取模型状态字典

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()  # 获取优化器状态字典

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()  # 获取学习率调度器状态字典

            # 保存 checkpoint
            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        """工具方法：如果存在检查点，则恢复模型。"""
        names = self.get_model_names()  # 获取所有模型名称
        
        # 遍历所有模型名称，检查是否存在检查点文件
        file_missing = False
        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True # 文件缺失
                break
        if file_missing: # 如果文件缺失，返回 0
            print("No checkpoint found, train from scratch")
            return 0
        print(f"Found checkpoint at {directory} (will resume training)")

        # 恢复模型
        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint( # 从检查点恢复
                path, self._models[name], self._optims[name], # 恢复模型、优化器
                self._scheds[name] # 恢复学习率调度器
            )
        return start_epoch # 返回开始的 epoch

    def load_model(self, directory, epoch=None):
        """工具方法：加载模型。"""
        if not directory: # 如果目录不存在，直接返回
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()  # 获取所有模型名称

        model_file = "model-best.pth.tar" # 默认情况下，加载最佳模型
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch) # 如果指定 epoch，加载指定 epoch 的模型

        # 遍历所有模型名称，加载模型
        for name in names: 
            model_path = osp.join(directory, name, model_file) # 模型路径

            if not osp.exists(model_path): # 如果模型路径不存在，抛出异常
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path) # 加载检查点
            state_dict = checkpoint["state_dict"] # 获取状态字典
            epoch = checkpoint["epoch"] # 获取 epoch
            val_result = checkpoint["val_result"] # 获取验证结果
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict) # 加载模型状态字典

    def set_model_mode(self, mode="train", names=None):
        """工具方法：设置模型模式 (train/eval)。"""
        names = self.get_model_names(names)  # 获取所有模型名称
        
        # 遍历所有模型名称，设置模型模式
        for name in names:
            if mode == "train":
                self._models[name].train()  # 设置模型为训练模式
            elif mode in ["test", "eval"]:
                self._models[name].eval()  # 设置模型为评估模式
            else:
                raise KeyError

    def update_lr(self, names=None):
        """工具方法：更新学习率。"""
        names = self.get_model_names(names)  # 获取所有模型名称

        # 遍历所有模型名称，更新学习率
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()  # 更新学习率

    def detect_anomaly(self, loss):
        """工具方法：检测损失是否为有限值。"""
        if not torch.isfinite(loss).all(): # 如果损失不是有限，抛出异常
            raise FloatingPointError("Loss is infinite or NaN!") 

    def init_writer(self, log_dir):
        """工具方法：初始化 TensorBoard。"""
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)  # 初始化 TensorBoard

    def close_writer(self):
        """工具方法：关闭 TensorBoard。"""
        if self._writer is not None:
            self._writer.close()  # 关闭 TensorBoard

    def write_scalar(self, tag, scalar_value, global_step=None):
        """工具方法：写入标量值到 TensorBoard。"""
        if self._writer is None:
            pass # 如果 writer 未初始化，则不执行任何操作
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)  # 写入标量值

    def train(self, start_epoch, max_epoch):
        """工具方法：通用训练循环。
            
            流程：(流程中全是抽象方法，需要子类实现)。
            1. 执行训练前的操作 before_train()
            2. 开始训练
                * 执行每个 epoch 前的操作 before_epoch()
                * 执行每个 epoch 的训练 run_epoch()
                * 执行每个 epoch 后的操作 after_epoch()
            6. 执行训练后的操作 after_train()
        """
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # 执行训练前的操作
        self.before_train()
        
        # 开始训练
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch() # 执行每个 epoch 前的操作
            self.run_epoch() # 执行每个 epoch 的训练
            self.after_epoch() # 执行每个 epoch 后的操作
        
        # 执行训练后的操作
        self.after_train() 

    def before_train(self):
        """训练前的操作。 (可选子类实现)"""
        pass

    def after_train(self):
        """训练后的操作。 (可选子类实现)"""
        pass

    def before_epoch(self):
        """每个 epoch 前的操作。 (可选子类实现)"""
        pass

    def after_epoch(self):
        """每个 epoch 后的操作。 (可选子类实现)"""
        pass

    def run_epoch(self):
        """执行每个 epoch 的训练。 (需要子类实现)"""
        raise NotImplementedError

    def test(self):
        """测试。 (需要子类实现)"""
        raise NotImplementedError

    def parse_batch_train(self, batch):
        """解析训练批次。 (需要子类实现)"""
        raise NotImplementedError

    def parse_batch_test(self, batch):
        """解析测试批次。 (需要子类实现)"""
        raise NotImplementedError

    def forward_backward(self, batch):
        """前向传播和反向传播。 (需要子类实现)"""
        raise NotImplementedError

    def model_inference(self, input):
        """模型推理。 (需要子类实现)"""
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        """工具方法：清零梯度。"""
        names = self.get_model_names(names)  # 获取所有模型名称
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()  # 清零梯度

    def model_backward(self, loss):
        """工具方法：模型根据 loss 的梯度反向传播。"""
        self.detect_anomaly(loss)  # 检测损失是否为有限值
        loss.backward()  # 反向传播

    def model_update(self, names=None):
        """工具方法：使用优化器更新模型参数。"""
        names = self.get_model_names(names)  # 获取所有模型名称
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()  # 更新模型参数

    def model_backward_and_update(self, loss, names=None):
        """工具方法：模型反向传播和更新。"""
        self.model_zero_grad(names)  # 清零梯度
        self.model_backward(loss)  # 反向传播
        self.model_update(names)  # 更新模型参数


class SimpleTrainer(TrainerBase):
    """一个实现通用功能的简单训练器类。"""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)  # 检查配置

        # 设置设备（GPU 或 CPU）
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # 保存一些常用变量为属性
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR
        self.cfg = cfg

        # 构建数据加载器、模型和评估器
        self.build_data_loader()  # 构建数据加载器
        self.build_model()  # 构建模型
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)  # 构建评估器
        
        # 初始化最佳结果
        self.best_result = -np.inf  

    def check_cfg(self, cfg):
        """检查配置中的某些变量是否正确设置（可选）。
        例如，一个训练器可能需要特定的采样器进行训练，如 'RandomDomainSampler'，
        因此可以进行检查：assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """通过配置构建数据管理器。并保存数据加载器、类别数、源域数等。"""
        dm = DataManager(self.cfg) # 通过配置创建数据管理器

        # 保存数据加载器（必选）
        self.train_loader_x = dm.train_loader_x # 有标签数据加载器
        self.train_loader_u = dm.train_loader_u  # 无标签数据加载器 (可选，可以为 None
        self.val_loader = dm.val_loader  # 验证数据加载器 (可选，可以为 None
        self.test_loader = dm.test_loader # 测试数据加载器
        
        # 保存类别数、源域数、类别名称字典（必选）
        self.num_classes = dm.num_classes # 类别数
        self.num_source_domains = dm.num_source_domains # 源域数
        self.lab2cname = dm.lab2cname  # 类别名称字典 {label: classname}

        # 保存数据管理器（可选）
        self.dm = dm 

    def build_model(self):
        """构建并注册模型。

        默认情况下构建一个 分类模型、优化器、调度器。

        如果需要，自定义训练器可以重新实现此方法。
        """
        cfg = self.cfg

        print("构建模型")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)  # 构建模型
        if cfg.MODEL.INIT_WEIGHTS: # 如果配置了预训练权重
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)  # 加载预训练权重
        self.model.to(self.device)  # 将模型移动到设备
        print(f"参数数量：{count_num_param(self.model):,}")
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)  # 构建优化器
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)  # 构建学习率调度器
        
        self.register_model("model", self.model, self.optim, self.sched)  # 注册模型

        device_count = torch.cuda.device_count() # GPU 数量
        if device_count > 1:
            print(f"检测到 {device_count} 个 GPU (使用 nn.DataParallel)")
            self.model = nn.DataParallel(self.model)  # 使用 DataParallel 并行化模型

    def before_train(self):
        """(实现父类的方法) 训练前的操作。"""
        # 设置输出目录
        if self.cfg.RESUME: # 如果配置了 RESUME
            directory = self.cfg.RESUME # 恢复 RESUME 目录
        else: # 否则按照配置的输出目录
            directory = self.cfg.OUTPUT_DIR

        # 如果存在检查点，则恢复模型
        self.start_epoch = self.resume_model_if_exist(directory)  

        # 初始化 summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir) # 创建日志目录
        self.init_writer(writer_dir) # 初始化 writer

        # 记录开始时间（用于计算经过的时间）
        self.time_start = time.time()

    def after_train(self):
        """(实现父类的方法) 训练后的操作。"""
        print("训练结束")

        # 如果需要测试，则测试，并保存最佳模型；否则保存最后一个 epoch 的模型
        do_test = not self.cfg.TEST.NO_TEST 
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("部署验证性能最好的模型")
                self.load_model(self.output_dir)
            else:
                print("部署最后一个 epoch 的模型")
            self.test()

        # 打印经过的时间
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"经过时间：{elapsed}")

        # 关闭 writer
        self.close_writer()

    def after_epoch(self):
        """(实现父类的方法) 每个 epoch 后的操作。"""
        last_epoch = (self.epoch + 1) == self.max_epoch # 是否是最后一个 epoch
        
        do_test = not self.cfg.TEST.NO_TEST # 是否需要验证（每个 epoch 后都验证，并保存验证性能最好的模型）
        
        meet_checkpoint_freq = (  # 是否满足保存检查点的频率
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # 保存模型
        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            # 如果每个 epoch 后都验证，则进行验证，并保存验证性能最好的模型
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )
        if meet_checkpoint_freq or last_epoch:
            # 如果满足保存检查点的频率或是最后一个 epoch，则保存模型
            self.save_model(self.epoch, self.output_dir)

    def train(self):
        """训练：（直接调用父类实现的通用 train 方法）"""
        super().train(self.start_epoch, self.max_epoch)  # 调用父类的 train 方法

    @torch.no_grad()
    def test(self, split=None):
        """测试：实现父类的抽象方法。"""
        self.set_model_mode("eval")
        
        self.evaluator.reset() # 重置评估器

        # 确定测试集（val or test，默认为测试集）
        if split is None: # 如果 split 为 None，则使用配置中的测试集
            split = self.cfg.TEST.SPLIT 
        if split == "val" and self.val_loader is not None: 
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        print(f"在 *{split}* 集上评估")

        # 开始测试
        for batch_idx, batch in enumerate(tqdm(data_loader)): # 遍历数据加载器
            input, label = self.parse_batch_test(batch) # 解析测试批次，获取输入和标签
            output = self.model_inference(input) # 模型推理
            self.evaluator.process(output, label) # 评估器评估模型输出和标签

        # 获取评估结果
        results = self.evaluator.evaluate() 
        for k, v in results.items(): # 打印评估结果
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0] # 返回第一个值：accuracy
    
    def model_inference(self, input):
        """模型推理：实现父类的抽象方法。"""
        return self.model(input) # 直接调用模型

    def parse_batch_test(self, batch):
        """解析测试批次：实现父类的抽象方法。"""
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        """获取当前学习率。"""
        names = self.get_model_names(names)
        name = names[0] # 只获取第一个模型的学习率
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """一个使用 有标签数据 和 无标签数据 的基础训练器类。

    在领域适应的背景下，有标签数据 和 无标签数据分别来自 源域 和 目标域。

    当涉及到 半监督学习 时，所有数据都来自 同一个域。
    """

    def run_epoch(self):
        """(实现父类的方法) 执行每个 epoch 的训练。"""
        self.set_model_mode("train")  # 设置模型为训练模式
        losses = MetricMeter()  # 初始化损失度量器
        batch_time = AverageMeter()  # 初始化批次时间度量器
        data_time = AverageMeter()  # 初始化数据加载时间度量器

        # 根据配置，判断迭代有标签数据集还是无标签数据集，确定 num_batches
        len_train_loader_x = len(self.train_loader_x)  # 有标签数据集的长度
        len_train_loader_u = len(self.train_loader_u)  # 无标签数据集的长度
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x  # 使用有标签数据集的长度
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u  # 使用无标签数据集的长度
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)  # 使用较小的长度
        else:
            raise ValueError  # 抛出异常

        # 数据集加载器
        train_loader_x_iter = iter(self.train_loader_x)  # 有标签数据集加载器
        train_loader_u_iter = iter(self.train_loader_u)  # 无标签数据集加载器

        # 开始迭代
        end = time.time()
        for self.batch_idx in range(self.num_batches):  # 遍历批次，采用 next() 获取下一个批次
            # 加载数据
            try:   # 获取下一个 有标签批次
                batch_x = next(train_loader_x_iter)
            except StopIteration: # 如果迭代完了，就重新初始化加载器
                train_loader_x_iter = iter(self.train_loader_x)  # 重新初始化加载器
                batch_x = next(train_loader_x_iter) # 获取下一个有标签批次
            try:  # 获取下一个 无标签批次
                batch_u = next(train_loader_u_iter)
            except StopIteration: # 如果迭代完了，就重新初始化加载器
                train_loader_u_iter = iter(self.train_loader_u)  # 重新初始化加载器
                batch_u = next(train_loader_u_iter) # 获取下一个无标签批次

            data_time.update(time.time() - end)  # 更新数据时间
            
            # 模型前向传播和反向传播
            loss_summary = self.forward_backward(batch_x, batch_u)  # 前向和反向传播
            batch_time.update(time.time() - end)  # 更新批次时间
            losses.update(loss_summary)  # 更新损失

            # 打印日志
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  # 是否满足打印频率
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ  # 是否只有少量批次
            if meet_freq or only_few_batches: # 如果满足打印频率或只有少量批次则打印日志
                # 根据剩余批次数估算剩余时间 eta
                nb_remain = 0 
                nb_remain += self.num_batches - self.batch_idx - 1  # 计算当前 epoch 剩余批次数
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches  # 计算后续 epoch 批次数
                eta_seconds = batch_time.avg * nb_remain  # 计算剩余时间
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                # 日志信息：epoch、batch、时间、数据加载时间、损失、学习率、剩余时间
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            n_iter = self.epoch * self.num_batches + self.batch_idx # 当前迭代次数
            for name, meter in losses.meters.items(): # 遍历多种损失
                self.write_scalar("train/" + name, meter.avg, n_iter)  # 记录 每种损失 到 TensorBoard
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)  # 记录 学习率 到 TensorBoard

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        """解析训练批次。"""

        # 获取有标签数据的图像和标签，无标签数据的图像
        input_x = batch_x["img"]  # 有标签数据的图像
        label_x = batch_x["label"]  # 有标签数据的标签
        input_u = batch_u["img"]  # 无标签数据的图像

        # 将数据移动到设备
        input_x = input_x.to(self.device) 
        label_x = label_x.to(self.device)  
        input_u = input_u.to(self.device)  

        return input_x, label_x, input_u  # 返回有标签数据的图像、标签和无标签数据的图像


class TrainerX(SimpleTrainer):
    """一个仅使用 有标签数据 的基础训练器类。"""

    def run_epoch(self):
        """(重写父类的方法) 执行每个 epoch 的训练。"""
        self.set_model_mode("train")  # 设置模型为训练模式
        losses = MetricMeter()  # 初始化损失度量器
        batch_time = AverageMeter()  # 初始化批次时间度量器
        data_time = AverageMeter()  # 初始化数据加载时间度量器
        self.num_batches = len(self.train_loader_x)  # 获取有标签数据集的批次数

        # 开始迭代
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x): # 遍历有标签数据集 train_loader_x
            data_time.update(time.time() - end)  # 更新数据加载时间
            
            # 前向和反向传播
            loss_summary = self.forward_backward(batch)  
            
            batch_time.update(time.time() - end)  # 更新批次时间
            losses.update(loss_summary)  # 更新损失

            # 打印日志
            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0  # 是否满足打印频率
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ  # 是否只有少量批次
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1  # 计算当前 epoch 剩余批次数
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches  # 计算后续 epoch 批次数
                eta_seconds = batch_time.avg * nb_remain  # 计算剩余时间
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                # 日志信息：epoch、batch、时间、数据加载时间、损失、学习率、剩余时间
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            
            n_iter = self.epoch * self.num_batches + self.batch_idx  # 当前迭代次数
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)  # 记录损失到 TensorBoard
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)  # 记录学习率到 TensorBoard

            end = time.time()

    def parse_batch_train(self, batch):
        """(重写父类的方法) 解析训练批次。"""
        input = batch["img"]  # 获取图像
        label = batch["label"]  # 获取标签
        domain = batch["domain"]  # 获取域标签

        input = input.to(self.device)  # 将图像移动到设备
        label = label.to(self.device)  # 将标签移动到设备
        domain = domain.to(self.device)  # 将域标签移动到设备

        return input, label, domain  # 返回图像、标签和域标签
