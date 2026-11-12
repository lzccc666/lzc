"""基于过去观测的稳健地面速度；不以类别代替运动，不读取未来。"""
from __future__ import annotations

import numpy as np


def robust_xy_speed(centers, times, *, minimum_pair_dt_s=.75):
    centers=np.asarray(centers,dtype=float)
    times=np.asarray(times,dtype=float)
    if len(centers)!=len(times) or len(times)<2 or np.any(np.diff(times)<=0):
        raise ValueError('运动观测需一一对应且时间严格递增')
    velocities=[]
    for i in range(len(times)):
        for j in range(i+1,len(times)):
            dt=times[j]-times[i]
            if dt>=minimum_pair_dt_s:
                velocities.append((centers[j,:2]-centers[i,:2])/dt)
    if not velocities:
        vector=(centers[-1,:2]-centers[-2,:2])/(times[-1]-times[-2])
    else:
        vector=np.median(velocities,axis=0)
    return float(np.linalg.norm(vector))
