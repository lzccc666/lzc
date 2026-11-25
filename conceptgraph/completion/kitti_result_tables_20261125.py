"""Derived class/micro/macro tables; no re-matching or altered denominators."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'outputs/results/kitti/validation/iterations/kitti_object_all_20261125_v3'
REPORT=ROOT/'outputs/reports/kitti_object_all_20261125'
PRIMARY=('car','truck','pedestrian')


def cell(value):
    tp=value['matches_3d'];gt=value['truth'];pred=value['predictions']
    return dict(TP=tp,GT=gt,predictions=pred,ignored_predictions=value['ignored_predictions'],
        no_depth=value['unknown_depth_predictions'],P_all_detections=tp/pred if pred else None,
        R=tp/gt if gt else None,F1_all_detections=2*tp/(gt+pred) if gt+pred else None,
        matched_center_error_m=value['matched_center_error_m'],
        conditional_2D_match_center_error_m=value['conditional_center_error_m'])


def main():
    source=RUN/'evaluation_summary.json';data=json.loads(source.read_text())
    if data['frames']!=7481:raise AssertionError('full_frame_denominator_required')
    methods={}
    for method,overall in data['methods'].items():
        classes={category:cell(data['per_class'][category][method]) for category in PRIMARY}
        macro={key:sum(r[key] for r in classes.values())/len(PRIMARY) for key in ('P_all_detections','R','F1_all_detections')}
        methods[method]=dict(micro=cell(overall),macro=macro,classes=classes,
            cyclist_bicycle_proxy_separate=cell(data['per_class']['bicycle'][method]))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),frames=7481,
        primary_GT=33191,car_GT_weight=27945/33191,methods=methods,
        metric='same class fullXYZ camera0 center distance<=3m, fixed1<GT_z<65m',
        P_and_F1='custom TP divided by all nonignored detections including no-depth; not official AP; unchanged legacy ignore protocol',
        macro='unweighted mean of the three primary class metrics; supplementary, does not replace micro33191GT',
        distance_appendix='independent global matching then assign GT z bins; no-depth predictions have no certified GT distance bin and must not be fabricated',
        source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        postprocessing_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_new_inference=True,no_label_or_matching_changes=True)
    with (REPORT/'full_class_micro_macro_tables_v1.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({m:dict(micro_R=r['micro']['R'],macro_R=r['macro']['R']) for m,r in methods.items()}))


if __name__=='__main__':main()
