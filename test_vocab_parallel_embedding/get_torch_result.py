import numpy as np
import torch
import torch.distributed as torch_dist
import base_class
import prepare_data
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

dim_1 = 56200
dim_2 = 4096
dim_3 = 12288

class TestTorch(base_class.BaseClass):
    def __init__(self, device, np_input_dir="./inputs_case1.npz", dtype="float32", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._device = device
        x_torch, table_torch, dout_torch = self._gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self._cal_torch_res(x_torch, table_torch, dout_torch)
        del x_torch 
        del table_torch 
        del dout_torch
        
        a = out_torch.cpu().detach().numpy()
        b = out_grads_torch.cpu().detach().numpy()
        np.savez(torch_dir, torch_out=a, torch_out_grad=b)
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()
    
    def _gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self._np_x,
            device=self._device,
            dtype=convert_dtype_to_torch_type("int64"),
        )
        table_torch = torch.tensor(
            self._np_table,
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
        return x_torch, table_torch, dout_torch

    def _cal_torch_res(self, x, table, dout):
        x_t = x
        table_t = table
        dout_t = dout

        if self._dtype == "bfloat16":
            table_t = table_t.to(dtype=torch.bfloat16)
            dout_t = dout_t.to(dtype=torch.bfloat16)

        embedding = torch.nn.Embedding(dim_1, dim_3, _weight=table_t, dtype=self._dtype)
        out = embedding(x_t)
        out_grads = torch.autograd.grad([out], [embedding.weight], grad_outputs=[dout_t])
        out_grads = out_grads[0]

        if self._dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = out_grads.to(dtype=torch.float32)
        
        return out, out_grads

dtype_list = ["float32", "float16", "bfloat16"]

prepare_data.generate_np_inputs_and_dout()

for dtype in dtype_list:

    np_input_dir = "./inputs_case1.npz"
    dtype = dtype
    torch_dir = "./torch_out_{dtype}.npz".format(dtype=dtype)

    test_torch = TestTorch('cuda', np_input_dir, dtype, torch_dir)
