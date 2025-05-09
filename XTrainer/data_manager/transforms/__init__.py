from .build import TRANSFORM_REGISTRY, build_train_transform, build_test_transform

<<<<<<< HEAD
from .base_class.TransformBase import TransformBase
=======
from .TransformBase import TransformBase
>>>>>>> 36fe5ca084dec516a944809acf4c7c0af6f81894
from .AutoAugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy, SubPolicy
from .RandomAugment import RandomIntensityAugment, ProbabilisticAugment
from .Cutout import Cutout
from .GaussianNoise import GaussianNoise
from .Random2DTranslation import Random2DTranslation

<<<<<<< HEAD
from .RandomResizedCrop import RandomResizedCrop  # 随机裁剪和缩放图像
from .RandomFlip import RandomFlip  # 随机水平翻转图像

=======
>>>>>>> 36fe5ca084dec516a944809acf4c7c0af6f81894
from .StandardNoAugTransform import StandardNoAugTransform  # 无增强的标准图像转换管道