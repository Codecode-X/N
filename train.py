import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.cocoop
import trainers.zsclip


def print_args(args, cfg):
    """ 打印参数 (args) 和配置 (cfg)。"""
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    """ 将参数 (args) 的设置覆盖到配置 (cfg)。"""
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains
    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    扩展配置 (cfg)。 
    COOP, COCOOP, 数据集的类子集的配置。
    示例：
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # 添加 COOP 训练器的配置
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # 上下文向量的数量
    cfg.TRAINER.COOP.CSC = False  # 是否使用类特定上下文
    cfg.TRAINER.COOP.CTX_INIT = ""  # 初始化词
    cfg.TRAINER.COOP.PREC = "fp16"  # 精度类型：fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 类标记位置：'middle', 'end', 'front'

    # 添加 COCOOP 训练器的配置
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # 上下文向量的数量
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # 初始化词
    cfg.TRAINER.COCOOP.PREC = "fp16"  # 精度类型：fp16, fp32, amp

    # 数据集的类子集配置
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # 类子集选项：all, base, new


def setup_cfg(args):
    """ 
    根据输入参数 设置配置，并 冻结配置 最后 返回配置。
    args 优先级 > 配置文件
    """
    # 获取默认配置
    cfg = get_cfg_default()
    # 扩展配置
    extend_cfg(cfg)

    # 1. 从数据集配置文件加载
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. 从方法配置文件加载
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. 从输入参数覆盖配置
    reset_cfg(cfg, args)

    # 4. 从可选输入参数覆盖配置
    cfg.merge_from_list(args.opts)

    # 冻结配置，防止后续修改
    cfg.freeze()

    return cfg


def main(args):
    # 设置配置
    cfg = setup_cfg(args)
    # 如果设置了随机种子，则固定种子
    if cfg.SEED >= 0:
        print("设置固定种子：{}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    # 设置日志记录器
    setup_logger(cfg.OUTPUT_DIR)

    # 如果支持 CUDA 且配置启用了 CUDA，则优化 CUDA 性能
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # 打印参数和配置
    print_args(args, cfg)
    print("收集环境信息 ...")
    print("** 系统信息 **\n{}\n".format(collect_env_info()))

    # 构建训练器
    trainer = build_trainer(cfg)

    # 如果仅评估模式，则加载模型并测试
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    # 如果不是仅评估模式，则进行训练
    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="数据集路径")
    parser.add_argument("--output-dir", type=str, default="", help="输出目录")
    parser.add_argument("--resume", type=str, default="", help="检查点目录（从该目录恢复训练）")
    parser.add_argument("--seed", type=int, default=-1, help="仅正值启用固定种子")
    parser.add_argument("--source-domains", type=str, nargs="+", help="DA/DG 的源域")
    parser.add_argument("--target-domains", type=str, nargs="+", help="DA/DG 的目标域")
    parser.add_argument("--transforms", type=str, nargs="+", help="数据增强方法")
    parser.add_argument("--config-file", type=str, default="", help="配置文件路径")
    parser.add_argument("--dataset-config-file", type=str, default="", help="数据集设置的配置文件路径")
    parser.add_argument("--trainer", type=str, default="", help="训练器名称")
    parser.add_argument("--backbone", type=str, default="", help="CNN 主干网络名称")
    parser.add_argument("--head", type=str, default="", help="头部名称")
    parser.add_argument("--eval-only", action="store_true", help="仅评估模式")
    parser.add_argument("--model-dir", type=str, default="", help="在仅评估模式下从此目录加载模型")
    parser.add_argument("--load-epoch", type=int, help="在评估时加载此 epoch 的模型权重")
    parser.add_argument("--no-train", action="store_true", help="不调用 trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="使用命令行修改配置选项")
    args = parser.parse_args()
    main(args)
