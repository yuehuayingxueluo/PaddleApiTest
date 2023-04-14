import numpy as np
import torch
import torch.distributed as torch_dist
import base_class
import prepare_data
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestTorch(base_class.BaseClass):
    def __init__(self, device, np_input_dir="./inputs_case1.npz", dtype="float32", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._device = device
        data_torch = self._gen_torch_inputs_and_dout()
        x_torch, weight_torch, bias_torch, dout_torch = data_torch[0], data_torch[1], data_torch[2], data_torch[3]
        out_torch, out_grads_torch = self._cal_torch_res(x_torch, weight_torch, bias_torch, dout_torch)
        del x_torch 
        del weight_torch
        del bias_torch
        del dout_torch
        
        a = out_torch.cpu().detach().numpy()
        b = [data.cpu().detach().numpy() for data in out_grads_torch]
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
        weight_torch = torch.tensor(
            self._np_weight,
            device=self._device,
            dtype=convert_dtype_to_torch_type(self._dtype)
            if self._dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        bias_torch = torch.tensor(
            self._np_bias,
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
        return [x_torch, weight_torch, bias_torch, dout_torch]

    def _cal_torch_res(self, x, weight, bias, dout):
        x_t = x
        weight_t = weight
        bias_t = bias
        dout_t = dout

        if self._dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            weight_t = weight.to(dtype=torch.bfloat16)
            bias_t = bias.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)

        linear = torch.nn.Linear(in_features=weight_t.shape[0], out_features=weight_t.shape[1])
        linear.weight = torch.nn.Parameter(weight_t.transpose(0, 1))
        linear.bias = torch.nn.Parameter(bias_t)
        out = linear(x_t)
        out_grads = torch.autograd.grad([out], [x_t], grad_outputs=[dout_t])

        if self._dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = [data.to(dtype=torch.float32) for data in out_grads]
        
        return out, out_grads

case_size = 5

dtype_list = ["float32", "float16", "bfloat16"]

prepare_data.generate_np_inputs_and_dout()

for case_id in range(case_size):
    for dtype_id, dtype in enumerate(dtype_list):

        np_input_dir = "./inputs_case{id_b}.npz".format(id_b=(case_id + 1))
        dtype = dtype
        torch_dir = "./{id}_torch_out_{dtype}.npz".format(id=case_id+1, dtype=dtype)

        test_torch = TestTorch('cuda', np_input_dir, dtype, torch_dir)
