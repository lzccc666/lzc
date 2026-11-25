"""CPU线程初始化幂等：spawn前并行工作前设interop1，重复同值保持。"""
import os,resource

def initialize_cpu_worker():
    import torch
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));torch.set_num_threads(2)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    def same_value_only(number):
        assert number==1 and torch.get_num_interop_threads()==1,'CPU runtime allows only frozen interop=1'
    torch.set_num_interop_threads=same_value_only
