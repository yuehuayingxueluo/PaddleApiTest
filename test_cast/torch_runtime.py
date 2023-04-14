import torch
from common import io
from common import framework
import numpy as np
import attribute

class Cast(framework.Framework):
  @staticmethod
  def convert_dtype(dtype):
    ret = None
    if dtype == "float32":
      ret = torch.float32
    elif dtype == "float16":
      ret = torch.float16
    elif dtype == "bfloat16":
      ret = torch.bfloat16
    elif dtype == "int64":
      ret = torch.int64
    elif dtype == "uint16":
      ret = torch.bfloat16
    return ret

  @staticmethod
  def launch_eager(input_data, attr):
    torch_src_dtype = Cast.convert_dtype(attr.src_dtype)
    torch_tgt_dtype = Cast.convert_dtype(attr.tgt_dtype)
    input_tensor_fp32 = torch.tensor(
      input_data.x,
      dtype=torch.float32,
      device=torch.device('cuda:0'),
    )
    t_tensor_fp32 = torch.tensor(
      input_data.out_t,
      dtype=torch.float32,
      device=torch.device('cuda:0'),
    )
    input_tensor_fp32.requires_grad = True
    input_tensor = input_tensor_fp32.to(torch_src_dtype)
    t_tensor = t_tensor_fp32.to(torch_tgt_dtype)
    out = input_tensor.to(torch_tgt_dtype)
    out_grads = torch.autograd.grad(
        [out], [input_tensor], [t_tensor]
    )
    return out.to(torch.float32).cpu().detach().numpy(), out_grads[0].to(torch.float32).cpu().detach().numpy()
