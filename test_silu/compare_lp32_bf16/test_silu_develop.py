import sys
import unittest

import numpy as np
import torch

import paddle
from paddle.utils import map_structure

sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

global_out = []
global_dout = []

class TestSiluDevelop():
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.init_threshold()
        self.init_np_inputs_and_dout()

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np.random.seed(2023)
        # init np array 
        self.np_x = np.random.random(size=self.shape).astype("float32") - 0.5
        self.np_dout = np.random.random(size=self.shape).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.nn.functional.silu(x)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager
        )
        del x_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        global_out.append(out_eager_np)
        global_dout.append(out_grads_eager_np)
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch


if __name__ == '__main__':
    shape_list = [[1, 8192, 4816]]
    for shape in shape_list:
        test_case_fp32 = TestSiluDevelop(shape, dtype="float32")
        test_case_fp32.test_eager_accuracy()
        test_case_bf16 = TestSiluDevelop(shape, dtype="bfloat16")
        test_case_bf16.test_eager_accuracy()
        try:
            np.testing.assert_array_equal(global_out[0], global_out[1])
            np.testing.assert_array_equal(global_dout[0], global_dout[1])
        except Exception as e:
            print(e)

