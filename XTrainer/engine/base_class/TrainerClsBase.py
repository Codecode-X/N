from ..build import TRAINER_REGISTRY
from .TrainerBase import TrainerBase
import torch
from tqdm import tqdm

@TRAINER_REGISTRY.register()
class TrainerClsBase(TrainerBase):
    """
    分类任务迭代训练器的基类。
    
    包含的方法：

    -------工具方法-------
    父类 TrainerClsBase:
        * init_writer: 初始化 TensorBoard。
        * close_writer: 关闭 TensorBoard。
        * write_scalar: 写入标量值到 TensorBoard。

        * register_model: 注册模型、优化器和学习率调度器。
        * get_model_names: 获取所有已注册的模型名称。

        * save_model: 保存模型，包括模型状态、epoch、优化器状态、学习率调度器状态、验证结果。
        * load_model: 加载模型，包括模型状态、epoch、验证结果。
        * resume_model_if_exist: 如果存在检查点，则恢复模型，包括模型状态、优化器状态、学习率调度器状态。

        * set_model_mode: 设置模型的模式 (train/eval)。

        * model_backward_and_update: 模型反向传播和更新，包括清零梯度、反向传播、更新模型参数。
        * update_lr: 调用学习率调度器的 step() 方法，更新 names 模型列表中的模型的学习率。
        * get_current_lr: 获取当前学习率。 

        * train: 通用训练循环，但里面包含的子方法 (before_train、after_train、before_epoch、
                after_epoch、run_epoch(必实现)) 需由子类实现。
    
    当前 TrainerBase:
        * None
    
    -------子类可重写的方法（可选）-------
    父类 TrainerClsBase:
        * check_cfg: 检查配置中的某些变量是否正确设置。 (未实现)

        * before_train: 训练前的操作。
        * after_train: 训练后的操作。
        * before_epoch: 每个 epoch 前的操作。 (未实现) 
        * after_epoch: 每个 epoch 后的操作。
        * run_epoch: 执行每个 epoch 的训练。
        * test: 测试方法。 (未实现) 
        * model_inference: 模型推理。
    
    当前 TrainerBase:
        * parse_batch_train 解析分类任务训练批次。(实现父类的方法)
        * parse_batch_test 解析测试任务训练批次。(实现父类的方法)
        * test: 测试方法。(实现父类的方法)

    -------需要子类重写的方法（必选）-------
    父类 TrainerClsBase:
        * init_model: 初始化模型，如冻结模型的某些层，加载预训练权重等。 (未实现 - 冻结模型某些层)
        * forward_backward: 前向传播和反向传播。
    
    当前 TrainerBase:
        * None
    """

    def parse_batch_train(self, batch):
        """
        (实现父类的方法) 解析分类任务训练批次。 
        此处直接从 batch 字典中获取输入图像、类别标签和域标签。

        参数：
            - batch (dict): 批次数据字典，包含输入图像和标签。

        返回：
            - input (Tensor): 输入图像。
            - label (Tensor): 类别标签。
        """
        input = batch["img"].to(self.device)  # 获取图像
        label = batch["label"].to(self.device)  # 获取标签 | shape: [batch, 1]
        return input, label  # 返回图像、标签


    def parse_batch_test(self, batch):
        """
        (实现父类的方法) 解析分类任务测试批次。
        返回解析得到的batch字典中的数据，例如分类问题的输入图像和类别标签。
        
        参数：
            - batch (dict): 批次数据字典，包含输入图像和标签。

        返回：
            - input (Tensor): 输入图像。
            - label (Tensor): 类别标签。
        """
        input = batch["img"].to(self.device)  # 获取图像
        label = batch["label"].to(self.device)  # 获取标签
        return input, label  # 返回图像、标签
    

    @torch.no_grad()
    def test(self, split=None):
        """
        测试。 (子类可重写)

        主要包括：
        * 设置模型模式为 eval, 重置评估器
        * 确定测试集 (val or test, 默认为测试集)
        * 开始测试
            - 遍历数据加载器
            - 解析测试批次，获取输入和标签 - self.parse_batch_test(batch)
            - 模型推理 - self.model_inference(input)
            - 评估器评估模型输出和标签 - self.evaluator.process(output, label)
        * 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        * 返回结果 (此处为 accuracy)
        """
        
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
        print(f"在 *{split}* 集上测试")

        # 开始测试
        for batch_idx, batch in enumerate(tqdm(data_loader)): # 遍历数据加载器
            input, label = self.parse_batch_test(batch) # 解析测试批次，获取输入和标签
            output = self.model_inference(input) # 模型推理
            self.evaluator.process(output, label) # 评估器评估模型输出和标签

        # 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        results = self.evaluator.evaluate() 
        for k, v in results.items(): 
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0] # 返回第一个值：accuracy