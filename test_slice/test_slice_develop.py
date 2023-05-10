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

class TestSliceDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(
            x_torch, dout_torch
        )
        del x_torch
        del dout_torch
        self.out_torch = map_structure(
            lambda x: x.cpu().detach().numpy(),
            out_torch,
        )
        self.out_grads_torch = map_structure(
            lambda x: x.cpu().numpy(),
            out_grads_torch,
        )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def tearDown(self):
        paddle.device.cuda.empty_cache()

    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 0

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[64, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dout_torch

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

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out_vec = []
        out_grads_vec = []
        for i in range(10):
            if hasattr(self,"is_last_slice") and self.is_last_slice:
                out = x[i+self.start_idx:]
            else:
                out = x[i+self.start_idx:i+self.start_idx+1]
            out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
            out_vec.append(out)
            out_grads_vec = out_grads_vec + list(out_grads)
        if self.dtype == "bfloat16":
            out_vec = map_structure(lambda x: x.to(dtype=torch.float32), out_vec)
            out_grads_vec = map_structure(lambda x: x.to(dtype=torch.float32), out_grads_vec)
        return out_vec, out_grads_vec

    def cal_eager_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out_vec = []
        out_grads_vec = []
        for i in range(10):
            if hasattr(self,"is_last_slice") and self.is_last_slice:
                out = x[i+self.start_idx:]
            else:
                out = x[i+self.start_idx:i+self.start_idx+1]
            out_grads = paddle.grad([out], [x], grad_outputs=[dout])
            out_vec.append(out)
            out_grads_vec = out_grads_vec + out_grads
        if self.dtype == "bfloat16":
            out_vec = map_structure(lambda x: paddle.cast(x,dtype="float32"), out_vec)
            out_grads_vec = map_structure(lambda x: paddle.cast(x,dtype="float32"), out_grads_vec)
        return out_vec, out_grads_vec

    def cal_static_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out_vec = []
        out_grads_vec = []
        for i in range(10):
            if hasattr(self,"is_last_slice") and self.is_last_slice:
                out = x[i+self.start_idx:]
            else:
                out = x[i+self.start_idx:i+self.start_idx+1]
            out_grads = paddle.static.gradients(
                [out], [x], target_gradients=[dout]
            )
            out_vec.append(out)
            out_grads_vec = out_grads_vec + out_grads
        if self.dtype == "bfloat16":
            out_vec = map_structure(lambda x: paddle.cast(x,dtype="float32"), out_vec)
            out_grads_vec = map_structure(lambda x: paddle.cast(x,dtype="float32"), out_grads_vec)
        return out_vec, out_grads_vec

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager
        )
        del x_eager
        del dout_eager
        out_eager_np = map_structure(
            lambda x: x.numpy(),
            out_eager,
        )
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
        # compare develop eager forward res with torch
        for idx in range(len(out_eager_np)):
            np_assert_accuracy(
                out_eager_np[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.slice",
            )
        # compare develop eager backward res with torch
        for idx in range(len(out_grads_eager_np)):
            np_assert_accuracy(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="backward",
                api="paddle.slice",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list= out_static + out_grads_static,
            )
            out_static, out_grads_static = out[:len(out_static)], out[len(out_static):]

        # compare develop static forward res with torch
        for idx in range(len(out_static)):
            np_assert_accuracy(
                out_static[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="forward",
                api="paddle.slice",
            )
        # compare develop static backward res with torch
        for idx in range(len(out_grads_static)):
            np_assert_accuracy(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="backward",
                api="paddle.slice",
            )

    def test_eager_stability(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dout_eager
        )
        out_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_eager_baseline,
        )
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dout_eager
            )
            out_eager = map_structure(
                lambda x: x.numpy(),
                out_eager,
            )
            out_grads_eager = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )
            # test develop eager forward stability
            for idx in range(len(out_eager)):
                np_assert_staility(
                    out_eager[idx],
                    out_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="paddle.slice",
                )
            # test develop eager backward stability
            for idx in range(len(out_grads_eager)):
                np_assert_staility(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="paddle.slice",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list= out_static_pg + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0:len(out_static_pg)], out[len(out_static_pg):]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
                    fetch_list= out_static_pg + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0:len(out_static_pg)], out[len(out_static_pg):]
                # test develop static forward stability
                for idx in range(len(out_static)):
                    np_assert_staility(
                        out_static[idx],
                        out_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="forward",
                        api="paddle.slice",
                    )
                # test develop static backward stability
                for idx in range(len(out_grads_static)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="paddle.slice",
                    )


class TestSliceDevelopCase1_FP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 0

class TestSliceDevelopCase1_BFP16(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 0

class TestSliceDevelopCase2_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 10

class TestSliceDevelopCase2_FP16(TestSliceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 10

class TestSliceDevelopCase2_BFP16(TestSliceDevelopCase2_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 10

class TestSliceDevelopCase3_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 20

class TestSliceDevelopCase3_FP16(TestSliceDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 20

class TestSliceDevelopCase3_BFP16(TestSliceDevelopCase3_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 20

class TestSliceDevelopCase4_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 30

class TestSliceDevelopCase4_FP16(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 30

class TestSliceDevelopCase4_BFP16(TestSliceDevelopCase4_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 30

class TestSliceDevelopCase5_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 40

class TestSliceDevelopCase5_FP16(TestSliceDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 40

class TestSliceDevelopCase5_BFP16(TestSliceDevelopCase5_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 40

class TestSliceDevelopCase6_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 50

class TestSliceDevelopCase6_FP16(TestSliceDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 50

class TestSliceDevelopCase6_BFP16(TestSliceDevelopCase6_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 50

class TestSliceDevelopCase7_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase7_FP16(TestSliceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase7_BFP16(TestSliceDevelopCase7_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase8_FP32(TestSliceDevelopCase1_FP32):
    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[64, 4096, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[1, 4096, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 0

class TestSliceDevelopCase8_FP16(TestSliceDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 0

class TestSliceDevelopCase8_BFP16(TestSliceDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 0

class TestSliceDevelopCase9_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 10

class TestSliceDevelopCase9_FP16(TestSliceDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 10

class TestSliceDevelopCase9_BFP16(TestSliceDevelopCase9_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 10

class TestSliceDevelopCase10_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 20

class TestSliceDevelopCase10_FP16(TestSliceDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 20

class TestSliceDevelopCase10_BFP16(TestSliceDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 20

class TestSliceDevelopCase11_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 30

class TestSliceDevelopCase11_FP16(TestSliceDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 30

class TestSliceDevelopCase11_BFP16(TestSliceDevelopCase11_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 30

class TestSliceDevelopCase12_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 40

class TestSliceDevelopCase12_FP16(TestSliceDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 40

class TestSliceDevelopCase12_BFP16(TestSliceDevelopCase12_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 40

class TestSliceDevelopCase13_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 50

class TestSliceDevelopCase13_FP16(TestSliceDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 50

class TestSliceDevelopCase13_BFP16(TestSliceDevelopCase13_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 50

class TestSliceDevelopCase14_FP32(TestSliceDevelopCase1_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase14_FP16(TestSliceDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase14_BFP16(TestSliceDevelopCase14_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.start_idx = 60
        self.is_last_slice = True

class TestSliceDevelopCase15_FP32(TestSliceDevelopCase10_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        # init np array 
        self.np_x = np.random.random(size=[4096, 1, 4096]).astype("float32") - 0.5
        self.np_dout = np.random.random(size=[2048, 1, 4096]).astype("float32") - 0.5
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout = self.np_dout.astype("float16")

    def cal_torch_res(self, x, dout):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout = dout.to(dtype=torch.bfloat16)
        out = x[0:2048]
        out_grads = torch.autograd.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dout, axes, starts, ends):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.slice(x, axes, starts, ends)
        out_grads = paddle.grad([out], [x], grad_outputs=[dout])
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dout, axes, starts, ends):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout = paddle.cast(dout, dtype="uint16")
        out = paddle.slice(x, axes, starts, ends)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype="float32"), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(
            x_eager, dout_eager, self.axes, self.starts, self.ends
        )
        del x_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager,
        )
        del out_eager
        del out_grads_eager
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
            api="paddle.slice",
        )
        # compare develop eager backward res with torch
        for idx in range(len(out_grads_eager_np)):
            np_assert_accuracy(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="eager",
                fwd_or_bkd="backward",
                api="paddle.slice",
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dout_static,
                    self.axes, self.starts, self.ends
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # compare develop static forward res with torch
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
            api="paddle.slice",
        )
        # compare develop static backward res with torch
        for idx in range(len(out_grads_static)):
            np_assert_accuracy(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                self.dtype,
                version_a="paddle_develop",
                version_b="torch",
                eager_or_static_mode="static",
                fwd_or_bkd="backward",
                api="paddle.slice",
            )

    def test_eager_stability(self):
        x_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(
            x_eager, dout_eager,self.axes, self.starts, self.ends
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
            lambda x: x.numpy(),
            out_grads_eager_baseline,
        )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()
        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(
                x_eager, dout_eager, self.axes, self.starts, self.ends
            )
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                lambda x: x.numpy(),
                out_grads_eager,
            )
            # test develop eager forward stability
            np_assert_staility(
                out_eager,
                out_eager_baseline_np,
                self.dtype,
                version="paddle_develop",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.slice",
            )
            # test develop eager backward stability
            for idx in range(len(out_grads_eager)):
                np_assert_staility(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="backward",
                    api="paddle.slice",
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    x_static,
                    dout_static,
                ) = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dout_static,
                    self.axes, self.starts, self.ends
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                # test develop static forward stability
                np_assert_staility(
                    out_static,
                    out_static_baseline,
                    self.dtype,
                    version="paddle_develop",
                    eager_or_static_mode="static",
                    fwd_or_bkd="forward",
                    api="paddle.slice",
                )
                # test develop static backward stability
                for idx in range(len(out_grads_static)):
                    np_assert_staility(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        self.dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="backward",
                        api="paddle.slice",
                    )


class TestSliceDevelopCase15_FP16(TestSliceDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]


class TestSliceDevelopCase15_BFP16(TestSliceDevelopCase15_FP32):
    def init_params(self):
        self.dtype = "bfloat16"
        self.axes = [0]
        self.starts = [0]
        self.ends = [2048]

if __name__ == '__main__':
    unittest.main()
