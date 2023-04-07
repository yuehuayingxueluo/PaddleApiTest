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

import unittest

import numpy as np
import torch
from utils import TOLERANCE, convert_dtype_to_torch_type

import paddle
from paddle.fluid.layers.utils import map_structure


class TestMatmulCase1(unittest.TestCase):
    def setUp(self):
        self.x_shape = [1, 32, 4096, 192]
        self.y_shape = [1, 32, 4096, 192]
        self.out_shape = [1, 32, 4096, 4096]
        self.transpose_x = False
        self.transpose_y = True
        self.dtypes = ["float32", "float16", "bfloat16"]
        self.places = ["gpu"]

    def cal_eager_res(self, x, y, transpose_x, transpose_y, dout, dtype):
        x_t = x
        y_t = y
        dout_t = dout
        if dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        out_grads = paddle.grad(
            [out], [x, y], grad_outputs=[dout_t], retain_graph=True
        )
        if dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, y, transpose_x, transpose_y, dout, dtype):
        x_t = x
        y_t = y
        dout_t = dout
        if dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            y_t = paddle.cast(y, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.matmul(x_t, y_t, transpose_x, transpose_y)
        out_grads = paddle.static.gradients(
            [out], [x, y], target_gradients=[dout_t]
        )
        if dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_torch_res(self, x, y, transpose_x, transpose_y, dout, dtype):
        x_t = x
        y_t = y
        dout_t = dout
        if dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            y_t = y.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        if transpose_x:
            x_t = torch.transpose(x_t, -1, -2)
        if transpose_y:
            y_t = torch.transpose(y_t, -1, -2)
        out = torch.matmul(x_t, y_t)
        out_grads = torch.autograd.grad([out], [x, y], grad_outputs=[dout_t])
        if dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def test_acc_and_stability(self):
        for place in self.places:
            for dtype in self.dtypes:
                atol = TOLERANCE[dtype]["atol"]
                rtol = TOLERANCE[dtype]["rtol"]
                # init numpy inputs and grad_outputs
                np_x = np.random.random(size=self.x_shape).astype(
                    "float32" if dtype == "bfloat16" else dtype
                )
                np_y = np.random.random(size=self.y_shape).astype(
                    "float32" if dtype == "bfloat16" else dtype
                )
                np_dout = np.random.random(size=self.out_shape).astype(
                    "float32" if dtype == "bfloat16" else dtype
                )
                # accuracy test
                # 1. calculate torch res
                x_torch = torch.tensor(
                    np_x,
                    device='cuda',
                    dtype=convert_dtype_to_torch_type(dtype)
                    if dtype != 'bfloat16'
                    else torch.float32,
                    requires_grad=True,
                )
                y_torch = torch.tensor(
                    np_y,
                    device='cuda',
                    dtype=convert_dtype_to_torch_type(dtype)
                    if dtype != 'bfloat16'
                    else torch.float32,
                    requires_grad=True,
                )
                dout_torch = torch.tensor(
                    np_dout,
                    device='cuda',
                    dtype=convert_dtype_to_torch_type(dtype)
                    if dtype != 'bfloat16'
                    else torch.float32,
                    requires_grad=True,
                )
                out_torch_acc, out_grads_torch_acc = self.cal_torch_res(
                    x_torch,
                    y_torch,
                    self.transpose_x,
                    self.transpose_y,
                    dout_torch,
                    dtype,
                )
                out_torch_acc = out_torch_acc.cpu().detach().numpy()
                out_grads_torch_acc = map_structure(
                    lambda x: x.cpu().numpy(),
                    out_grads_torch_acc,
                )
                # 2. calculate eager res and compare with torch
                x_eager = paddle.to_tensor(
                    np_x,
                    dtype=dtype if dtype != 'bfloat16' else "float32",
                    place=place,
                )
                x_eager.stop_gradient = False
                y_eager = paddle.to_tensor(
                    np_y,
                    dtype=dtype if dtype != 'bfloat16' else "float32",
                    place=place,
                )
                y_eager.stop_gradient = False
                dout_eager = paddle.to_tensor(
                    np_dout,
                    dtype=dtype if dtype != 'bfloat16' else "float32",
                    place=place,
                )
                dout_eager.stop_gradient = False
                out_eager_acc, out_grads_eager_acc = self.cal_eager_res(
                    x_eager,
                    y_eager,
                    self.transpose_x,
                    self.transpose_y,
                    dout_eager,
                    dtype,
                )
                out_eager_acc = out_eager_acc.numpy()
                out_grads_eager_acc = map_structure(
                    lambda x: x.numpy(), out_grads_eager_acc
                )
                # compare eager res with torch
                try:
                    np.testing.assert_allclose(
                        out_eager_acc,
                        out_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare matmul eager forward res with torch failed in %s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare matmul eager forward res with torch succeed in %s'
                        % dtype
                    )
                try:
                    for idx in range(len(out_grads_eager_acc)):
                        np.testing.assert_allclose(
                            out_grads_eager_acc[idx],
                            out_grads_torch_acc[idx],
                            atol,
                            rtol,
                            err_msg=(
                                'compare matmul eager grad res with torch failed in %s'
                            )
                            % dtype,
                        )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare matmul eager grad res with torch succeed in %s'
                        % dtype
                    )
                # 3. calculate static res and compare with torch
                paddle.enable_static()
                mp, sp = paddle.static.Program(), paddle.static.Program()
                with paddle.static.program_guard(mp, sp):
                    x_static = paddle.static.data(
                        'x',
                        shape=self.x_shape,
                        dtype=dtype if dtype != "bfloat16" else "float32",
                    )
                    x_static.stop_gradient = False
                    y_static = paddle.static.data(
                        'y',
                        shape=self.y_shape,
                        dtype=dtype if dtype != "bfloat16" else "float32",
                    )
                    y_static.stop_gradient = False
                    dout_static = paddle.static.data(
                        'dout',
                        shape=self.out_shape,
                        dtype=dtype if dtype != "bfloat16" else "float32",
                    )
                    dout_static.stop_gradient = False
                    (out_static_pg, out_grads_static_pg,) = self.cal_static_res(
                        x_static,
                        y_static,
                        self.transpose_x,
                        self.transpose_y,
                        dout_static,
                        dtype,
                    )
                exe = paddle.static.Executor(
                    place=paddle.CUDAPlace(0)
                    if place == "gpu"
                    else paddle.CPUPlace()
                )
                exe.run(sp)
                out = exe.run(
                    mp,
                    feed={"x": np_x, "y": np_y, "dout": np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static_acc, out_grads_static_acc = out[0], out[1:]
                paddle.disable_static()

                # compare static res with torch
                try:
                    np.testing.assert_allclose(
                        out_static_acc,
                        out_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare matmul static forward res with torch failed in %s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare matmul static forward res with torch succeed in %s'
                        % dtype
                    )
                try:
                    for idx in range(len(out_grads_static_acc)):
                        np.testing.assert_allclose(
                            out_grads_static_acc[idx],
                            out_grads_torch_acc[idx],
                            atol,
                            rtol,
                            err_msg=(
                                'compare matmul static grad res with torch failed%s'
                            )
                            % dtype,
                        )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare matmul static grad res with torch succeed in %s'
                        % dtype
                    )
                # test stability
                eager_stability = True
                static_stability = True
                torch_stability = True
                for n in range(1000):
                    # 1.test eager res stability
                    if eager_stability:
                        out_eager, out_grads_eager = self.cal_eager_res(
                            x_eager,
                            y_eager,
                            self.transpose_x,
                            self.transpose_y,
                            dout_eager,
                            dtype,
                        )
                        out_eager = out_eager.numpy()
                        out_grads_eager = map_structure(
                            lambda x: x.numpy(), out_grads_eager
                        )
                        try:
                            np.testing.assert_equal(
                                out_grads_eager,
                                out_grads_eager_acc,
                                err_msg=(
                                    'paddle.matmul eager forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            eager_stability = False
                            print(error)
                        try:
                            for idx in range(len(out_grads_eager)):
                                np.testing.assert_equal(
                                    out_grads_eager[idx],
                                    out_grads_eager_acc[idx],
                                    err_msg=(
                                        'paddle.matmul eager grad is unstable in%s'
                                    )
                                    % dtype,
                                )
                        except AssertionError as error:
                            eager_stability = False
                            print(error)
                    # 2. test static res stability
                    if static_stability:
                        out = exe.run(
                            mp,
                            feed={"x": np_x, "y": np_y, "dout": np_dout},
                            fetch_list=[out_static_pg] + out_grads_static_pg,
                        )
                        out_static, out_grads_static = out[0], out[1:]
                        try:
                            np.testing.assert_equal(
                                out_static,
                                out_static_acc,
                                err_msg=(
                                    'paddle.matmul static forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            static_stability = False
                            print(error)
                        try:
                            for idx in range(len(out_grads_static)):
                                np.testing.assert_equal(
                                    out_grads_static[idx],
                                    out_grads_static_acc[idx],
                                    err_msg=(
                                        'paddle.matmul static grad is unstable in %s'
                                    )
                                    % dtype,
                                )
                        except AssertionError as error:
                            static_stability = False
                            print(error)
                    # 3. test torch res stability
                    if torch_stability:
                        out_torch, out_grads_torch = self.cal_torch_res(
                            x_torch,
                            y_torch,
                            self.transpose_x,
                            self.transpose_y,
                            dout_torch,
                            dtype,
                        )
                        out_torch = out_torch.cpu().detach().numpy()
                        out_grads_torch = map_structure(
                            lambda x: x.cpu().numpy(),
                            out_grads_torch,
                        )
                        try:
                            np.testing.assert_equal(
                                out_torch,
                                out_torch_acc,
                                err_msg=(
                                    'torch.matmul forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            torch_stability = False
                            print(error)
                        try:
                            for idx in range(len(out_grads_torch)):
                                np.testing.assert_equal(
                                    out_grads_torch[idx],
                                    out_grads_torch_acc[idx],
                                    err_msg=(
                                        'torch.matmul grad is unstable in %s'
                                    )
                                    % dtype,
                                )
                        except AssertionError as error:
                            torch_stability = False
                            print(error)


class TestMatmulCase2(TestMatmulCase1):
    def setUp(self):
        self.x_shape = [1, 32, 4096, 4096]
        self.y_shape = [1, 32, 4096, 4096]
        self.out_shape = [1, 32, 4096, 4096]
        self.transpose_x = False
        self.transpose_y = False
        self.dtypes = ["float32", "float16", "bfloat16"]
        self.places = ["gpu"]


class TestMatmulCase3(TestMatmulCase1):
    def setUp(self):
        self.x_shape = [1, 32, 4096, 4096]
        self.y_shape = [1, 32, 4096, 192]
        self.out_shape = [1, 32, 4096, 4096]
        self.transpose_x = False
        self.transpose_y = True
        self.dtypes = ["float32", "float16", "bfloat16"]
        self.places = ["gpu"]


if __name__ == '__main__':
    unittest.main()
