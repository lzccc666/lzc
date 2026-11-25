"""CARLA新实验的可逆存储：原RGB/原深度PNG+打包mask，避免重复保存crop像素。"""
import gzip,pickle
from pathlib import Path
import numpy as np
from PIL import Image

def crop_boxes(xyxy,width,height):
 boxes=[]
 for row in xyxy:
  x1,y1,x2,y2=row
  boxes.append([float(x1-min(20,x1)),float(y1-min(20,y1)),float(x2+min(20,width-x2)),float(y2+min(20,height-y2))])
 return boxes

def save_compact(folder,gobs,rgb):
 folder=Path(folder);mask=np.asarray(gobs['mask']);packed={k:v for k,v in gobs.items() if k not in {'image_crops','mask'}}
 packed['storage_schema']='carla_lossless_crop_reference_v1';packed['mask_shape']=list(mask.shape);packed['mask_dtype']=mask.dtype.str;packed['mask_packbits']=np.packbits(mask.astype(bool).reshape(-1));packed['crop_boxes_xyxy']=crop_boxes(gobs['xyxy'],rgb.shape[1],rgb.shape[0]);packed['crop_source']='rgb.png';packed['crop_padding_pixels']=20
 # crop恢复逐像素核对；存储策略不改变本次实际输入、SAM mask或CLIP特征。
 original=Image.fromarray(rgb)
 assert len(packed['crop_boxes_xyxy'])==len(gobs['image_crops'])
 for box,crop in zip(packed['crop_boxes_xyxy'],gobs['image_crops']):assert np.array_equal(np.asarray(original.crop(box)),np.asarray(crop))
 p=folder/'detections_compact.pkl.gz'
 with gzip.open(p,'xb',compresslevel=6) as f:pickle.dump(packed,f,protocol=4)
 restored=load_compact(p)
 for key in ['mask','xyxy','class_id','confidence','image_feats','text_feats']:assert np.array_equal(np.asarray(restored[key]),np.asarray(gobs[key])),key
 return p

def load_compact(path):
 path=Path(path)
 with gzip.open(path,'rb') as f:g=pickle.load(f)
 if g.pop('storage_schema')!='carla_lossless_crop_reference_v1':raise ValueError('unsupported evidence schema')
 shape=g.pop('mask_shape');dtype=np.dtype(g.pop('mask_dtype'));g['mask']=np.unpackbits(g.pop('mask_packbits'),count=int(np.prod(shape))).reshape(shape).astype(dtype)
 rgb=Image.open(path.parent/g.pop('crop_source')).convert('RGB');g['image_crops']=[rgb.crop(box) for box in g.pop('crop_boxes_xyxy')];g.pop('crop_padding_pixels');return g
