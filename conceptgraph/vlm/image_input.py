"""训练和推理共用的Qwen图像预处理，保留小crop像素并补足patch尺寸。"""
from __future__ import annotations

import math
from PIL import Image

IMAGE_PREPROCESS_VERSION = 'qwen-rgb-maxside-graypad28-v1'


def prepare_qwen_image(image: Image.Image, max_image_side: int | None) -> Image.Image:
    rgb=image.convert('RGB')
    if max_image_side is not None:
        if max_image_side < 28:
            raise ValueError('Qwen图像上限必须至少为28像素')
        if max(rgb.size)>max_image_side:
            scale=max_image_side/max(rgb.size)
            rgb=rgb.resize((max(1,round(rgb.width*scale)),max(1,round(rgb.height*scale))),Image.Resampling.LANCZOS)
    width=max(28,rgb.width,math.ceil(rgb.height/200))
    height=max(28,rgb.height,math.ceil(rgb.width/200))
    if (width,height)!=rgb.size:
        padded=Image.new('RGB',(width,height),(127,127,127))
        padded.paste(rgb,((width-rgb.width)//2,(height-rgb.height)//2))
        rgb=padded
    return rgb
