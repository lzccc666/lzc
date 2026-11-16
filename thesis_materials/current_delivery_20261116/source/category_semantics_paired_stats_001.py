"""从逐场景整数分子/分母重算六条件的11日志配对差异；用新输出ID复跑。"""
import argparse,csv
from pathlib import Path
import numpy as np
from conceptgraph.completion.stage9 import dump,digest
R=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();source=R/'nuscenes_ablation/category_semantics_001/scene_counts.csv';O=R/'nuscenes_ablation'/args.run_id;O.mkdir(exist_ok=False);rows=list(csv.DictReader(source.open()));logs=sorted({p['log'] for p in rows});conditions=list(dict.fromkeys(p['condition'] for p in rows));metrics=[k[:-10] for k in rows[0] if k.endswith('_numerator')];draw=np.random.default_rng(17).integers(0,len(logs),(4000,len(logs)));aggregate={}
 for condition in conditions:
  aggregate[condition]={}
  for metric in metrics:
   a=np.asarray([[sum(int(p[metric+'_numerator']) for p in rows if p['condition']==condition and p['log']==log),sum(int(p[metric+'_denominator']) for p in rows if p['condition']==condition and p['log']==log)] for log in logs]);b=a[draw].sum(1);aggregate[condition][metric]=(a[:,0].sum()/a[:,1].sum(),b[:,0]/b[:,1])
 output=[]
 for condition in conditions:
  for metric in metrics:
   value,boot=aggregate[condition][metric];full,full_boot=aggregate['full'][metric];low,high=np.percentile(boot-full_boot,[2.5,97.5]);output.append(dict(condition=condition,metric=metric,absolute_delta_vs_full=value-full,relative_change_vs_full=(value-full)/full if full else None,paired_ci95_low=low,paired_ci95_high=high,paired_logs=len(logs),bootstrap_repeats=4000))
 with (O/'paired_differences.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
 dump(O/'receipt.json',dict(source_sha256=digest(source),script_sha256=digest(__file__),same_logs_and_denominators=True,bootstrap_seed=17,repetitions=4000,units=11,goal_complete=False))
if __name__=='__main__':main()
