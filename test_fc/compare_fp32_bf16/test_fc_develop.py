
import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
sys.path.append("..")
from utils import (
    TOLERANCE,
    convert_dtype_to_torch_type,
    np_assert_accuracy,
    np_assert_staility,
)

global_out = []
global_dout = []

class TestFCDevelop(unittest.TestCase):
    def __init__(self, shape, dtype):
        self.dtype = dtype
        self.shape = shape  
        self.init_threshold()
        self.init_np_inputs_and_dout()
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):

        self.np_x = np.random.random(size=self.shape["x"]).astype("float32")
        self.np_w = np.random.random(size=self.shape["w"]).astype("float32")
        self.np_b = np.random.random(size=self.shape["b"]).astype("float32")
        self.np_dout = np.random.random(size=self.shape["dout"]).astype("float32")

        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_w = self.np_w.astype("float16")
            self.np_b = self.np_b.astype("float16")
            self.np_dout = self.np_dout.astype("float16")   
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        w_eager = paddle.to_tensor(
            self.np_w,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        w_eager.stop_gradient = False
        b_eager = paddle.to_tensor(
            self.np_b,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        b_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, w_eager, b_eager, dout_eager

    def cal_eager_res(self, x, w, b, dout):
        x_t = x
        w_t = w
        b_t = b
        dout_t = dout
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            w_t = paddle.cast(w, dtype="uint16")
            b_t = paddle.cast(b, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.incubate.nn.functional.fused_linear(x_t, w_t, b_t)
        out_grads = paddle.grad(
            [out], [x, w, b], grad_outputs=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, w_eager, b_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, w_eager, b_eager, dout_eager)
        del x_eager
        del w_eager 
        del b_eager
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

if __name__ == '__main__':
    x = [[1, 8192, 14336], [1, 8192, 14336], [1, 8192, 1792], [1, 8192, 4816]]
    w = [[14336, 5376], [14336, 9632], [1792, 14336], [4816, 14336]]
    b = [[5376], [9632], [14336], [14336]]
    dout = [[1, 8192, 5376], [1, 8192, 9632], [1, 8192, 14336], [1, 8192, 14336]]
    shape = {}
    shape_list = []
    for i in range(len(x)):
        shape["x"] = x[i - 1]
        shape["w"] = w[i - 1]
        shape["b"] = b[i - 1]
        shape["dout"] = dout[i - 1]
        shape_list.append(shape)
    for shape in shape_list:
        global_out.clear()
        global_dout.clear()
        print("shape: ", shape)
        test_case_fp32 = TestFCDevelop(shape, dtype="float32")
        test_case_fp32.test_eager_accuracy()
        test_case_bf16 = TestFCDevelop(shape, dtype="bfloat16")
        test_case_bf16.test_eager_accuracy()
        try:
            np.testing.assert_array_equal(global_out[0], global_out[1])
        except Exception as e:
            print(e)

        try:
            np.testing.assert_array_equal(global_dout[0], global_dout[1])
        except Exception as e:
            print(e)
