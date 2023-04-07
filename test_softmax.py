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


class TestSoftMaxCase1(unittest.TestCase):
    def setUp(self):
        self.input_shape = [1, 32, 4096, 4096]
        self.out_shape = [1, 32, 4096, 4096]
        self.use_cudnn = True
        self.axis = -1
        self.dtypes = ["bfloat16"]
        self.places = ["gpu"]

    def cal_eager_res(self,input,use_cudnn,axis,dout, dtype):
        input_t = input
        dout_t = dout
        if dtype == "bfloat16":
            input_t = paddle.cast(input, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.fluid.layers.softmax(input_t, use_cudnn=use_cudnn, axis=axis)
        out_grads = paddle.grad(
            [out], [input], grad_outputs=[dout_t], retain_graph=True
        )
        if dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads[0]

    def cal_static_res(self,input,use_cudnn,axis,dout, dtype):
        input_t = input
        dout_t = dout
        if dtype == "bfloat16":
            input_t = paddle.cast(input, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.fluid.layers.softmax(input_t, use_cudnn=use_cudnn, axis=axis)
        out_grads = paddle.static.gradients(
            [out], [input], target_gradients=[dout_t]
        )
        if dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads[0]

    def cal_torch_res(self,input,axis,dout,dtype):
        input_t = input
        dout_t = dout
        if dtype == "bfloat16":
            input_t = input_t.to(dtype=torch.bfloat16)
            dout_t = dout_t.to(dtype=torch.bfloat16)
        out = torch.nn.functional.softmax(input_t, axis)
        out_grads = torch.autograd.grad([out], [input], grad_outputs=[dout_t])
        if dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads[0]

    def test_acc_and_stability(self):
        for place in self.places:
            for dtype in self.dtypes:
                atol = TOLERANCE[dtype]["atol"]
                rtol = TOLERANCE[dtype]["rtol"]
                # init numpy inputs and grad_outputs
                np_input = np.random.random(size=self.input_shape).astype(
                    "float32" if dtype == "bfloat16" else dtype
                )
                np_dout = np.random.random(size=self.out_shape).astype(
                    "float32" if dtype == "bfloat16" else dtype
                )
                # accuracy test
                # 1. calculate torch res
                input_torch = torch.tensor(
                    np_input,
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
                    input_torch,
                    -1,
                    dout_torch,
                    dtype,
                )
                out_torch_acc = out_torch_acc.cpu().detach().numpy()
                out_grads_torch_acc = out_grads_torch_acc.cpu().numpy()
                # 2. calculate eager res and compare with torch
                input_eager = paddle.to_tensor(
                    np_input,
                    dtype=dtype if dtype != 'bfloat16' else "float32",
                    place=place,
                )
                input_eager.stop_gradient = False
                dout_eager = paddle.to_tensor(
                    np_dout,
                    dtype=dtype if dtype != 'bfloat16' else "float32",
                    place=place,
                )
                dout_eager.stop_gradient = False
                out_eager_acc, out_grads_eager_acc = self.cal_eager_res(
                    input_eager,
                    self.use_cudnn,
                    -1,
                    dout_eager,
                    dtype,
                )
                out_eager_acc = out_eager_acc.numpy()
                out_grads_eager_acc = out_grads_eager_acc.numpy()
                # compare eager res with torch
                try:
                    np.testing.assert_allclose(
                        out_eager_acc,
                        out_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare softmax eager forward res with torch failed in %s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare softmax eager forward res with torch succeed in %s'
                        % dtype
                    )
                try:
                    np.testing.assert_allclose(
                        out_grads_eager_acc,
                        out_grads_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare softmax eager grad res with torch failed in %s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare softmax eager grad res with torch succeed in %s'
                        % dtype
                    )
                # 3. calculate static res and compare with torch
                paddle.enable_static()
                mp, sp = paddle.static.Program(), paddle.static.Program()
                with paddle.static.program_guard(mp, sp):
                    input_static = paddle.static.data(
                        'input',
                        shape=self.input_shape,
                        dtype=dtype if dtype != "bfloat16" else "float32",
                    )
                    input_static.stop_gradient = False
                    dout_static = paddle.static.data(
                        'dout',
                        shape=self.out_shape,
                        dtype=dtype if dtype != "bfloat16" else "float32",
                    )
                    dout_static.stop_gradient = False
                    (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                        input_static,
                        self.use_cudnn,
                        -1,
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
                    feed={"input": np_input, "dout": np_dout},
                    fetch_list=[out_static_pg] + [out_grads_static_pg],
                )
                out_static_acc, out_grads_static_acc = out[0], out[1]
                paddle.disable_static()

                # compare static res with torch
                try:
                    np.testing.assert_allclose(
                        out_static_acc,
                        out_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare softmax static forward res with torch failed in %s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare softmax static forward res with torch succeed in %s'
                        % dtype
                    )
                try:
                    np.testing.assert_allclose(
                        out_grads_static_acc,
                        out_grads_torch_acc,
                        atol,
                        rtol,
                        err_msg=(
                            'compare softmax static grad res with torch failed%s'
                        )
                        % dtype,
                    )
                except AssertionError as error:
                    print(error)
                else:
                    print(
                        'compare softmax static grad res with torch succeed in %s'
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
                            input_eager,
                            self.use_cudnn,
                            -1,
                            dout_eager,
                            dtype,
                        )
                        out_eager = out_eager.numpy()
                        out_grads_eager = out_grads_eager.numpy()
                        try:
                            np.testing.assert_equal(
                                out_grads_eager,
                                out_grads_eager_acc,
                                err_msg=(
                                    'paddle.fluid.layers.softmax eager forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            eager_stability = False
                            print(error)
                        try:
                            np.testing.assert_equal(
                                out_grads_eager,
                                out_grads_eager_acc,
                                err_msg=(
                                    'paddle.fluid.layers.softmax eager grad is unstable in%s'
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
                            feed={"input": np_input, "dout": np_dout},
                            fetch_list=[out_static_pg] + [out_grads_static_pg],
                        )
                        out_static, out_grads_static = out[0], out[1]
                        try:
                            np.testing.assert_equal(
                                out_static,
                                out_static_acc,
                                err_msg=(
                                    'paddle.fluid.layers.softmax static forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            static_stability = False
                            print(error)
                        try:
                            np.testing.assert_equal(
                                out_grads_static,
                                out_grads_static_acc,
                                err_msg=(
                                    'paddle.fluid.layers.softmax static grad is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            static_stability = False
                            print(error)
                    # 3. test torch res stability
                    if torch_stability:
                        out_torch, out_grads_torch = self.cal_torch_res(
                            input_torch,
                            -1,
                            dout_torch,
                            dtype,
                        )
                        out_torch = out_torch.cpu().detach().numpy()
                        out_grads_torch = out_grads_torch.cpu().numpy()
                        try:
                            np.testing.assert_equal(
                                out_torch,
                                out_torch_acc,
                                err_msg=(
                                    'torch.nn.functional.softmax forward is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            torch_stability = False
                            print(error)
                        try:
                            np.testing.assert_equal(
                                out_grads_torch,
                                out_grads_torch_acc,
                                err_msg=(
                                    'torch.nn.functional.softmax grad is unstable in %s'
                                )
                                % dtype,
                            )
                        except AssertionError as error:
                            torch_stability = False
                            print(error)


if __name__ == '__main__':
    unittest.main()
