from data_manager.transforms import TRANSFORM_REGISTRY
from .base_class.TransformBase import TransformBase
"""
Source: https://github.com/DeepVoltaire/AutoAugment
"""
import numpy as np
import random
from PIL import Image, ImageOps, ImageEnhance

__all__ = ["ImageNetPolicy", "CIFAR10Policy", "SVHNPolicy"]


@TRANSFORM_REGISTRY.register()
class ImageNetPolicy(TransformBase):

    def __init__(self, cfg):
        self.fillcolor = cfg.INPUT.ImageNetPolicy.fillcolor \
            if hasattr(cfg.INPUT.ImageNetPolicy, 'fillcolor') else (128, 128, 128)
        
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, self.fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, self.fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, self.fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, self.fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, self.fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, self.fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, self.fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, self.fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, self.fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, self.fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, self.fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, self.fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, self.fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, self.fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, self.fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, self.fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, self.fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, self.fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, self.fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, self.fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, self.fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, self.fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, self.fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, self.fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, self.fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

@TRANSFORM_REGISTRY.register()
class CIFAR10Policy(TransformBase):

    def __init__(self, cfg):
        self.fillcolor = cfg.INPUT.CIFAR10Policy.fillcolor \
            if hasattr(cfg.INPUT.CIFAR10Policy, 'fillcolor') else (128, 128, 128)
        
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, self.fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, self.fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, self.fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, self.fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, self.fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, self.fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, self.fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, self.fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, self.fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, self.fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, self.fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, self.fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, self.fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, self.fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, self.fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, self.fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, self.fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, self.fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, self.fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, self.fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, self.fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, self.fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, self.fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, self.fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, self.fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

@TRANSFORM_REGISTRY.register()
class SVHNPolicy(TransformBase):

    def __init__(self, cfg):
        self.fillcolor=cfg.INPUT.SVHNPolicy.fillcolor \
            if hasattr(cfg.INPUT.SVHNPolicy, 'fillcolor') else (128, 128, 128)
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, self.fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, self.fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, self.fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, self.fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, self.fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, self.fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, self.fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, self.fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, self.fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, self.fillcolor),
            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, self.fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, self.fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, self.fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, self.fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, self.fillcolor),
            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, self.fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, self.fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, self.fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, self.fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, self.fillcolor),
            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, self.fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, self.fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, self.fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, self.fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, self.fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),  
            "shearY": np.linspace(0, 0.3, 10),  
            "translateX": np.linspace(0, 150 / 331, 10),  
            "translateY": np.linspace(0, 150 / 331, 10),  
            "rotate": np.linspace(0, 30, 10),  
            "color": np.linspace(0.0, 0.9, 10),  
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),  
            "solarize": np.linspace(256, 0, 10),  
            "contrast": np.linspace(0.0, 0.9, 10),  
            "sharpness": np.linspace(0.0, 0.9, 10),  
            "brightness": np.linspace(0.0, 0.9, 10),  
            "autocontrast": [0] * 10,  
            "equalize": [0] * 10,  
            "invert": [0] * 10,  
        }

        
        
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128, ) * 4), rot
            ).convert(img.mode)

        
        func = {
            "shearX":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (
                    1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0,
                    1, 0
                ),
                fillcolor=fillcolor,
            ),
            "translateY":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (
                    1, 0, 0, 0, 1, magnitude * img.size[1] * random.
                    choice([-1, 1])
                ),
                fillcolor=fillcolor,
            ),
            "rotate":
            lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color":
            lambda img, magnitude: ImageEnhance.Color(img).
            enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize":
            lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize":
            lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast":
            lambda img, magnitude: ImageEnhance.Contrast(img).
            enhance(1 + magnitude * random.choice([-1, 1])),
            "sharpness":
            lambda img, magnitude: ImageEnhance.Sharpness(img).
            enhance(1 + magnitude * random.choice([-1, 1])),
            "brightness":
            lambda img, magnitude: ImageEnhance.Brightness(img).
            enhance(1 + magnitude * random.choice([-1, 1])),
            "autocontrast":
            lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize":
            lambda img, magnitude: ImageOps.equalize(img),
            "invert":
            lambda img, magnitude: ImageOps.invert(img),
        }

        
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img