# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

class TestAssignDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch = self.gen_torch_inputs_and_dout()
        out_torch = self.cal_torch_res(x_torch)
        del x_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        del out_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[2048 , 1 , 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32
        )
        return x_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        return x_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        return x_static

    def cal_torch_res(self, x):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
        y = x
        if self.dtype == "bfloat16":
            y = y.to(dtype=torch.float32)
        return y

    def cal_eager_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        y= paddle.assign(x)
        if self.dtype == "bfloat16":
            y = paddle.cast(y, dtype="float32")
        return y

    def cal_static_res(self, x):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
        y = paddle.assign(x)
        if self.dtype == "bfloat16":
            y = paddle.cast(y, dtype="float32")
        return y


    def cal_static_res_with_feed(self):    
         with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self.gen_static_inputs_and_dout()
                y_static = self.cal_static_res(x_static)
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x},
                fetch_list=[y_static],
            )
            y = out[0]
            return y
    

    def test_eager_accuracy(self):
        x_eager = self.gen_eager_inputs_and_dout()
        out_eager = self.cal_eager_res(x_eager)
        del x_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()
        # compare develop eager forward res with torch
        np_assert_accuracy(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="eager",
            fwd_or_bkd="forward",
            api="paddle.assign",
        )    

    def test_eager_stability(self):
        x_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline = self.cal_eager_res(x_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager = self.cal_eager_res(x_eager)
            out_eager = out_eager.numpy()
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.assign",
            )

    def test_static_accuracy(self):   
        out_static = self.cal_static_res_with_feed()
        np_assert_accuracy(
        out_static,
        self.out_torch,
        self.atol,
        self.rtol,
        self.dtype,
        version_a="paddle_develop",
        version_b="torch",
        eager_or_static_mode="static",
        fwd_or_bkd="forward",
        api="paddle.assign",
    )    

    def test_static_statbility(self):   
        out_static_baseline =  self.cal_static_res_with_feed()
        for i in range(50):
            out_static = self.cal_static_res_with_feed()
            np_assert_accuracy(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            self.dtype,
            version_a="paddle_develop",
            version_b="torch",
            eager_or_static_mode="static",
            fwd_or_bkd="forward",
            api="paddle.assign",
        )    


class TestAssignDevelopCase1_FP16(TestAssignDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestAssignDevelopCase1_BFP16(TestAssignDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"


class TestAssignDevelopCase2_FP32(TestAssignDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"

    def init_np_inputs_and_dout(self):
        # init np array
        self.np_x = np.random.random(size=[4096 ,1 ,4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")


class TestAssignDevelopCase2_FP16(TestAssignDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"


class TestAssignDevelopCase2_BFP16(TestAssignDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"

if __name__ == '__main__':
    np.random.seed(2023)
    unittest.main()