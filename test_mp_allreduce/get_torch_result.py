import numpy as np
import torch
import torch.distributed as torch_dist
import init_config_class
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestTorch(init_config_class.InitClass):
    def __init__(self, group, device, np_input_dir="./inputs_case1.npz", dtype="float32", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        self._device = device
        x_torch, dout_torch = self._gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self._cal_torch_res(x_torch, dout_torch)
        del x_torch 
        del dout_torch
        local_rank = torch.distributed.get_rank()
        if local_rank == 0:
            a = out_torch.cpu().detach().numpy()
            b = out_grads_torch.cpu().detach().numpy()
            np.savez(torch_dir, torch_out=a, torch_out_grad=b)
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
    
    def _gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self._np_x,
            device=self._device,
            dtype=convert_dtype_to_torch_type(self._dtype)
            if self._dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self._np_dout,
            device=self._device,
            dtype=convert_dtype_to_torch_type(self._dtype)
            if self._dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dout_torch

    def _cal_torch_res(self, x, dout):
        out = x
        if self._dtype == "bfloat16":
            out = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        
        torch_dist.all_reduce(out, group=group)

        if self._dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            dout = dout.to(dtype=torch.float32)
        
        return out, dout

dtype_list = ["float32", "float16", "bfloat16"]

torch_dist.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
world_size = torch_dist.get_world_size()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
group = torch_dist.new_group([i for i in range(world_size)])

for case_id in range(2):
    for dtype in dtype_list:

        np_input_dir = "./inputs_case{id}.npz".format(id=(case_id + 1))
        torch_dir = "{case_id}_torch_out_{dtype}.npz".format(case_id=case_id+1, dtype=dtype)

        test_torch = TestTorch(group ,device, np_input_dir, dtype, torch_dir)
