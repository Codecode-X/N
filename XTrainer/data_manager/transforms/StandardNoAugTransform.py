from data_manager.transforms import TRANSFORM_REGISTRY
<<<<<<< HEAD
from .base_class.TransformBase import TransformBase
=======
from .TransformBase import TransformBase
>>>>>>> 36fe5ca084dec516a944809acf4c7c0af6f81894
from utils import standard_image_transform

@TRANSFORM_REGISTRY.register()
class StandardNoAugTransform(TransformBase):
    """
    标准化的无增强图像转换管道
    (resize + center_crop + to_rgb + to_tensor + normalize)

    主要步骤：
        - 将图像保持长宽比缩放到指定大小，然后中心裁剪到指定大小
        - 将图像转换为 RGB 格式
    """

    def __init__(self, cfg):
        self.target_size = cfg.INPUT.SIZE
        assert isinstance(self.target_size, int), "cfg.INPUT.SIZE 必须是单个整数"
        
        # 转换方法
        self.transform = standard_image_transform(self.target_size, 
                                                  interp_mode=cfg.INPUT.INTERPOLATION)

    def __call__(self, img):
        return self.transform(img)
