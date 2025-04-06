from ..build import TRAINER_REGISTRY
from .TrainerBase import TrainerBase
import torch

@TRAINER_REGISTRY.register()
class TrainerMcqBase(TrainerBase):
    """
    分类任务迭代训练器的基类。
    
    包含的方法：

    -------工具方法-------
    父类 TrainerMcqBase:
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
    父类 TrainerMcqBase:
        * check_cfg: 检查配置中的某些变量是否正确设置。 (未实现)

        * before_train: 训练前的操作。
        * after_train: 训练后的操作。
        * before_epoch: 每个 epoch 前的操作。 (未实现) 
        * after_epoch: 每个 epoch 后的操作。
        * run_epoch: 执行每个 epoch 的训练。
        * model_inference: 模型推理。
    
    当前 TrainerBase:
        * parse_batch_train 解析分类任务训练批次。(实现父类的方法)
        * parse_batch_test 解析测试任务训练批次。(实现父类的方法)

    -------需要子类重写实现的方法（必选）-------
    父类 TrainerMcqBase:
        * init_model: 初始化模型，如冻结模型的某些层，加载预训练权重等。 (未实现 - 冻结模型某些层)
        * forward_backward: 前向传播和反向传播。
    
    当前 TrainerBase:
        * test: 测试方法。
    """

    def parse_batch_train(self, batch):
        """
        (实现父类的方法) 解析分类任务训练批次。 
        此处直接从 batch 字典中获取输入图像、类别标签和域标签。

        参数：
            - batch (dict): 批次数据字典，包含：
                - 输入图像、答案选项数量、答案选项、正确答案索引、正确答案类型。
                
        返回：
            - input (Tensor): 输入图像 | [batch, 3, 224, 224]。
            - num_choices (Tensor): 答案选项数量 | [batch]。
            - choices (Tensor): 所有答案选项文本 | [batch, num_choices]。
            - correct_answer (Tensor): 正确答案选项的索引 | [batch]。
            - correct_answer_type (Tensor): 正确答案选项的类型 | [batch]。 
        """
        input = batch["img"].to(self.device)  # 获取一个批次的图像 | [batch, 3, 224, 224]
        num_choices = batch["num_choices"].to(self.device) # 获取答案选项数量 | [batch]
        choices = batch["choices"].to(self.device) # 获取所有答案选项文本 | [batch, num_choices]
        correct_answer = batch["correct_answer"].to(self.device) # 获取正确答案选项的索引 | [batch]
        correct_answer_type = batch["correct_answer_type"].to(self.device) # 获取正确答案选项的类型 | [batch]
        return input, num_choices, choices, correct_answer, correct_answer_type

    def parse_batch_test(self, batch):
        """
        (实现父类的方法) 解析分类任务测试批次。
        返回解析得到的batch字典中的数据，例如分类问题的输入图像和类别标签。
        
        参数：
            - batch (dict): 批次数据字典，包含输入图像和标签。
                - 输入图像、答案选项数量、答案选项、正确答案索引、正确答案类型。

        返回：
            - input (Tensor): 输入图像 | [batch, 3, 224, 224]。
            - num_choices (Tensor): 答案选项数量 | [batch]。
            - choices (Tensor): 所有答案选项文本 | [batch, num_choices]。
            - correct_answer (Tensor): 正确答案选项的索引 | [batch]。
            - correct_answer_type (Tensor): 正确答案选项的类型 | [batch]。 
        """
        input = batch["img"].to(self.device)  # 获取一个批次的图像 | [batch, 3, 224, 224]
        num_choices = batch["num_choices"].to(self.device) # 获取答案选项数量 | [batch]
        choices = batch["choices"].to(self.device) # 获取所有答案选项文本 | [batch, num_choices]
        
        correct_answer = batch["correct_answer"].to(self.device) # 获取正确答案选项的索引 | [batch]
        correct_answer_type = batch["correct_answer_type"].to(self.device) # 获取正确答案选项的类型 | [batch]
        return input, num_choices, choices, correct_answer, correct_answer_type
    

    @torch.no_grad()
    def test(self, split=None):
        """
        测试。 (需要子基类根据任务特点重写)

        主要包括：
        * 设置模型模式为 eval, 重置评估器
        * 确定测试集 (val or test, 默认为测试集)
        * 开始测试
            - 遍历数据加载器
            - 解析测试批次，获取输入和GT
            - 模型推理
            - 评估器评估模型输出和GT
        * 使用 evaluator 对结果进行评估，并将结果记录在 tensorboard
        * 返回结果 (此处为 accuracy)
        """
        raise NotImplementedError("TrainerMcqBase中test()方法需要子基类根据模型特点实现")
        