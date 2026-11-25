"""Regression for the real zero-box SAM crash and persistent YOLO class setup."""

from types import SimpleNamespace

import numpy as np
from PIL import Image


def test_empty_detection_frame_is_saved_and_reused_models_keep_class_setup(
    tmp_path, monkeypatch
):
    import gzip
    import pickle
    import torch
    from conceptgraph.scripts import streamlined_detections as module

    color = tmp_path / "frame.jpg"
    Image.new("RGB", (10, 8), "black").save(color)
    classes = tmp_path / "classes.txt"
    classes.write_text("car\ntruck\n")

    class Dataset:
        color_paths = [str(color)]

        def __len__(self):
            return 1

    class Detector:
        setups = 0

        def set_classes(self, names):
            self.setups += 1

        def predict(self, *args, **kwargs):
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        xyxy=torch.empty((0, 4)),
                        conf=torch.empty(0),
                        cls=torch.empty(0),
                    )
                )
            ]

    class SAM:
        def predict(self, *args, **kwargs):
            raise AssertionError("empty boxes must not enter SAM")

    monkeypatch.setattr(module, "get_dataset", lambda **kwargs: Dataset())
    monkeypatch.setattr(module, "save_hydra_config", lambda *args: None)
    detector = Detector()
    cfg = SimpleNamespace(
        dataset_config="unused",
        start=0,
        end=-1,
        stride=1,
        dataset_root=str(tmp_path),
        scene_id="scene",
        desired_height=8,
        desired_width=10,
        device="cpu",
        classes_file=str(classes),
        exp_suffix="streamlined_yolo_outdoor_objects_v1",
        save_video=False,
    )
    (tmp_path / "scene").mkdir()
    models = (
        detector,
        SAM(),
        SimpleNamespace(visual=SimpleNamespace(output_dim=4)),
        None,
        None,
    )
    module.run_detections(cfg, models)
    module.run_detections(cfg, models)
    assert detector.setups == 1
    with gzip.open(
        tmp_path
        / "scene/exp_streamlined_yolo_outdoor_objects_v1/detections/frame.pkl.gz",
        "rb",
    ) as handle:
        saved = pickle.load(handle)
    assert saved["mask"].shape == (0, 8, 10)
    assert saved["image_feats"].shape == (0, 4)
    assert saved["image_crops"] == []
    assert np.isfinite(saved["text_feats"]).all()


def test_sam_inplace_resize_cannot_corrupt_yolo_boxes_or_crops(tmp_path, monkeypatch):
    import gzip
    import pickle
    import torch
    from conceptgraph.scripts import streamlined_detections as module

    color=tmp_path/'frame.jpg'
    Image.new('RGB',(100,80),'white').save(color)
    classes=tmp_path/'classes.txt'
    classes.write_text('car\n')
    authoritative=torch.tensor([[20.,10.,70.,60.]])

    class Dataset:
        color_paths=[str(color)]

        def __len__(self):
            return 1

    class Detector:
        def set_classes(self,names):
            pass

        def predict(self,*args,**kwargs):
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=authoritative,
                conf=torch.tensor([.9]),cls=torch.tensor([0])))]

    class SAM:
        def predict(self,*args,bboxes,**kwargs):
            bboxes.mul_(.64)
            return [SimpleNamespace(masks=SimpleNamespace(data=torch.ones((1,80,100))))]

    captured=[]

    def features(image,detections,*args):
        captured.append(detections.xyxy.copy())
        return [Image.new('RGB',(50,50))],np.ones((1,4)),np.ones((1,4))

    monkeypatch.setattr(module,'get_dataset',lambda **kwargs:Dataset())
    monkeypatch.setattr(module,'save_hydra_config',lambda *args:None)
    monkeypatch.setattr(module,'compute_clip_features_batched',features)
    monkeypatch.setattr(module,'vis_result_fast',lambda image,*args:(image,None))
    (tmp_path/'scene').mkdir()
    cfg=SimpleNamespace(dataset_config='unused',start=0,end=-1,stride=1,
        dataset_root=str(tmp_path),scene_id='scene',desired_height=80,desired_width=100,
        device='cpu',classes_file=str(classes),exp_suffix='streamlined_yolo_outdoor_objects_v1',
        save_video=False)
    module.run_detections(cfg,(Detector(),SAM(),None,None,None))
    assert torch.equal(authoritative,torch.tensor([[20.,10.,70.,60.]]))
    assert np.array_equal(captured[0],authoritative.numpy())
    with gzip.open(tmp_path/'scene/exp_streamlined_yolo_outdoor_objects_v1/detections/frame.pkl.gz','rb') as f:
        saved=pickle.load(f)
    assert saved['frame_provenance']['bbox_coordinate_frame']=='original_rgb_pixels'
    assert np.array_equal(saved['xyxy'],authoritative.numpy())

    from conceptgraph.slam.utils import filter_gobs
    filtered=filter_gobs(SimpleNamespace(mask_area_threshold=1,skip_bg=True,
        max_bbox_area_ratio=1.0,mask_conf_threshold=0.1), saved, np.ones((80,100,3)))
    assert filtered['frame_provenance']==saved['frame_provenance']
    assert len(filtered['xyxy'])==1
