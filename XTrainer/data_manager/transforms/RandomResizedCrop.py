from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
from torchvision.transforms import RandomResizedCrop as _RandomResizedCrop
from torchvision.transforms import InterpolationMode

@TRANSFORM_REGISTRY.register()
class RandomResizedCrop(TransformBase):
    """
    随机裁剪和缩放图像
    """

    def __init__(self, cfg):
        self.target_size = cfg.INPUT.SIZE
        self.s_ = cfg.INPUT.RandomResizedCrop.scale # 随机裁剪的比例范围
        interp_mode = getattr(InterpolationMode, cfg.INPUT.INTERPOLATION.upper(), InterpolationMode.BILINEAR) # 获取插值模式
        
        assert isinstance(self.target_size, int), "cfg.INPUT.SIZE 必须是单个整数"
        
        # 转换方法
        self.transform = _RandomResizedCrop(self.target_size, scale=self.s_, interpolation=interp_mode)

    def __call__(self, img):
        return self.transform(img)
