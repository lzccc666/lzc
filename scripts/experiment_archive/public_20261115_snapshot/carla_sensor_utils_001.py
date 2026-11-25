import numpy as np
class CameraSpec:
    intrinsic=np.array([[320.,0,320],[0,320.,180],[0,0,1]])

def observed_mask_color(rgb, mask):
    """简单颜色基线，仅读取当前SAM前景像素；不是人工或场景GT标签。"""
    import cv2
    pixels = rgb[np.asarray(mask,dtype=bool)]
    if len(pixels) < 20:
        return 'unknown',{}
    hsv = cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(-1,3)
    h,s,v = hsv.T
    counts = dict(red=int((((h<12)|(h>165))&(s>90)&(v>45)).sum()),
                  blue=int(((h>90)&(h<135)&(s>90)&(v>45)).sum()),
                  white=int(((s<50)&(v>150)).sum()))
    name = max(counts,key=counts.get)
    return (name if counts[name]/len(pixels) >= .12 else 'unknown'),dict(
        counts=counts,pixels=len(pixels),source='current RGB + actual SAM mask; deterministic HSV baseline')
