import torch
from .build import EVALUATOR_REGISTRY
from .EvaluatorBase import EvaluatorBase
from collections import OrderedDict, defaultdict

@ EVALUATOR_REGISTRY.register()
class EvaluatorMcq(EvaluatorBase):
    """
    接口类 评估器。
    
    子类需要实现以下方法：
        - __init__：初始化评估器。
        - reset：重置评估器状态。
        - process：处理模型输出和真实标签。
        - evaluate：计算评估结果并返回。
    """

    def __init__(self, cfg, dm):
        """ 
        初始化MCQ任务评估器。
        
        参数:
            - cfg (CfgNode): 配置。
            - dm (Dataset): 数据集管理器。
        """
        super().__init__(cfg, dm)

        self.cfg = cfg
        split = cfg.TEST.SPLIT 
        if split == "val" and dm.val_loader is not None: 
            data_loader = dm.val_loader
        elif split == "test" and dm.test_loader is not None:
            data_loader = dm.test_loader
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")

        if "syn" in self.cfg.DATASET.NAME.lower(): # 根据是否为合成数据集，映射错误答案索引到答案类型
            print("=> Evaluating on synthetic dataset")
            self.wrong_answer_to_type = {0: 'positive', 1: 'hybrid', 2: 'hybrid', 3: 'negative'}
        else:
            self.wrong_answer_to_type = {1: 'hybrid', 2: 'positive', 3: 'negative'}
        
        # 初始化字典以跟踪每种类型的正确答案和总问题数
        self.correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        # 初始化错误答案计数
        self.wrong_answer_counts = {k: 0 for k in self.wrong_answer_to_type.keys()}
        # 初始化每种类型的预测计数
        self.predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        # 初始化嵌套字典以映射每种问题类型到每种错误答案类型的计数
        self.wrong_answers_by_question_type = {
            'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0}
        }

        self.total_questions = 0 # 初始化数据集中问题的总数
        self.correct_answers_sum = 0 # 初始化正确答案计数

    def reset(self):
        """重置评估器状态。"""
        self.correct_answers_sum = 0
        self.total_questions = 0
        self.correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.wrong_answer_counts = {k: 0 for k in self.wrong_answer_to_type.keys()}
        self.predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.wrong_answers_by_question_type = {
            'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0}
        }

    def process(self, logits, correct_answer, correct_answer_type):
        """处理模型输出和真实标签。
        参数：
            - logits (torch.Tensor): 模型输出 [batch, num_classes]
            - labels (torch.LongTensor): 真实标签 [batch]
            - correct_answer (torch.LongTensor): 正确答案索引 [batch]
            - correct_answer_type (list<str>): 正确答案类型 (batch)
        """
        # 更新总问题数
        self.total_questions += logits.size(0) # [batch_size]=[100]
        
        # 计算当前批次的正确预测数
        preds = logits.argmax(dim=1).cpu().numpy() # [batch_size]=[100]
        labels = torch.tensor(correct_answer, dtype=torch.long).cpu().numpy() # [batch_size]=[100]
        correct_predictions = (preds == labels).sum().item()

        # 累加正确答案计数
        self.correct_answers_sum += correct_predictions

        # 更新每种答案类型的计数并跟踪预测
        batch_size = logits.size(0)
        for i in range(batch_size):
            answer_type = correct_answer_type[i]  # 当前问题的答案类型
            self.total_questions_by_type[answer_type] += 1
            
            if preds[i] == correct_answer[i]: # 如果预测正确
                self.correct_answers_by_type[answer_type] += 1
                self.predictions_by_type[answer_type] += 1
            else: # 如果预测错误
                wrong_answer_idx = preds[i] # 获取错误答案的类型
                wrong_answer_type = self.wrong_answer_to_type.get(wrong_answer_idx, "unknown_type")
                self.wrong_answer_counts[wrong_answer_idx] += 1
                self.predictions_by_type[wrong_answer_type] += 1
                self.wrong_answers_by_question_type[answer_type][wrong_answer_type] += 1
    

    def evaluate(self):
        """计算评估结果并返回。"""
        total_accuracy = self.correct_answers_sum / self.total_questions # 计算总体准确率

        # 计算每种类型的准确率
        positive_accuracy = self.correct_answers_by_type['positive'] / self.total_questions_by_type['positive'] if self.total_questions_by_type['positive'] > 0 else float('nan')
        negative_accuracy = self.correct_answers_by_type['negative'] / self.total_questions_by_type['negative'] if self.total_questions_by_type['negative'] > 0 else float('nan')
        hybrid_accuracy = self.correct_answers_by_type['hybrid'] / self.total_questions_by_type['hybrid'] if self.total_questions_by_type['hybrid'] > 0 else float('nan')

        # 找出最常见的错误答案类型
        most_common_wrong_answer_type = max(self.wrong_answer_counts, key=self.wrong_answer_counts.get)

        # 计算每种错误答案类型的百分比
        total_wrong_answers = sum(self.wrong_answer_counts.values())
        wrong_answer_percentages = {self.wrong_answer_to_type[k]: (v / total_wrong_answers) * 100 for k, v in self.wrong_answer_counts.items()}

        # 打印并返回包含所有计算指标的有序字典
        results = OrderedDict({
            'total_accuracy': total_accuracy,
            'positive_accuracy': positive_accuracy,
            'negative_accuracy': negative_accuracy,
            'hybrid_accuracy': hybrid_accuracy,
            'most_common_wrong_answer_type': self.wrong_answer_to_type[most_common_wrong_answer_type],
            'wrong_answer_percentages': wrong_answer_percentages,
            'predictions_by_type': self.predictions_by_type,
            'wrong_answers_by_question_type': self.wrong_answers_by_question_type
        })
        if self.cfg.VERBOSE:
            print( # 打印评估结果
                "=> result\n"
                f"Total Accuracy: {total_accuracy:.4f}\n"
                f"Positive Accuracy: {positive_accuracy:.4f}\n"
                f"Negative Accuracy: {negative_accuracy:.4f}\n"
                f"Hybrid Accuracy: {hybrid_accuracy:.4f}\n"
                f"Most Common Wrong Answer Type: {self.wrong_answer_to_type[most_common_wrong_answer_type]}\n"
                f"Wrong Answer Percentages: {wrong_answer_percentages}\n"
                f"Predictions by Type: {self.predictions_by_type}\n"
                f"Wrong Answers by Question Type: {self.wrong_answers_by_question_type}\n"
            )
        return results