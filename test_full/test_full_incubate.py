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

import paddle

sys.path.append("..")
from utils import TOLERANCE


class TestFullIncubateCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_shape = np_inputs_array["shape"]
        self.np_fill_value = np_inputs_array["fill_value"]
        self.np_dtype = np_inputs_array["dtype"]

    def gen_eager_inputs(self):
        shape_eager = paddle.to_tensor(self.np_shape)
        fill_value_eager = self.np_fill_value
        dtype_eager = self.dtype
        return shape_eager, fill_value_eager, dtype_eager

    def gen_static_inputs(self):
        shape_static = paddle.to_tensor(self.np_shape)
        fill_value_static = self.np_fill_value
        dtype_static = self.dtype
        return shape_static, fill_value_static, dtype_static

    def cal_eager_res(self, shape, fill_value, dtype):
        out = paddle.fluid.layers.fill_constant(shape, dtype, fill_value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def cal_static_res(self, shape, fill_value, dtype):
        out = paddle.fluid.layers.fill_constant(shape, dtype, fill_value)
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out

    def test_eager_accuracy(self):
        # get develop eager res
        develop_res_array = np.load(self.save_eager_res_path)
        out_eager_develop = develop_res_array["out_eager"]

        # calculate incubate eager res
        shape_eager, fill_value_eager, dtype_eager = self.gen_eager_inputs()
        out_eager = self.cal_eager_res(
            shape_eager, fill_value_eager, dtype_eager
        )
        del shape_eager
        del fill_value_eager
        del dtype_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        del out_eager
        paddle.device.cuda.empty_cache()

        # compare incubate eager res with develop eager res
        np.testing.assert_equal(
            out_eager_np,
            out_eager_develop,
            err_msg=(
                'Incubate: compare fill_constant incubate eager forward res with develop eager forward res failed in %s dtype'
            )
            % self.dtype,
        )

    def test_static_accuracy(self):
        # get develop static res
        develop_res_array = np.load(self.save_static_res_path)
        out_static_develop = develop_res_array["out_static"]

        # calculate incubate static res
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    shape_static,
                    fill_value_static,
                    dtype_static,
                ) = self.gen_static_inputs()
                out_static = self.cal_static_res(
                    shape_static,
                    fill_value_static,
                    dtype_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                fetch_list=[out_static],
            )
            out_static = out[0]

        # compare incubate static res with develop static res
        np.testing.assert_equal(
            out_static,
            out_static_develop,
            err_msg=(
                'Incubate: compare fill_constant incubate static forward res with develop static forward res failed in %s dtype'
            )
            % self.dtype,
        )

    def test_eager_stability(self):
        shape_eager, fill_value_eager, dtype_eager = self.gen_eager_inputs()
        out_eager_baseline = self.cal_eager_res(
            shape_eager, fill_value_eager, dtype_eager
        )
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager = self.cal_eager_res(
                shape_eager, fill_value_eager, dtype_eager
            )
            out_eager = out_eager.numpy()
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Incubate: paddle.full eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                (
                    shape_static,
                    fill_value_static,
                    dtype_static,
                ) = self.gen_static_inputs()
                out_static_pg = self.cal_static_res(
                    shape_static,
                    fill_value_static,
                    dtype_static,
                )
            exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
            exe.run(sp)
            out = exe.run(
                mp,
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            for i in range(50):
                out = exe.run(
                    mp,
                    fetch_list=[out_static_pg],
                )
                out_static = out[0]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Incubate: paddle.full static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )


class TestFullIncubateCase1_FP16(TestFullIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestFullIncubateCase1_BF16(TestFullIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


class TestFullIncubateCase2_FP32(TestFullIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"


class TestFullIncubateCase2_FP16(TestFullIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.bias_after_scale = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"


class TestFullIncubateCase2_BF16(TestFullIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.bias_after_scale = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"


if __name__ == '__main__':
    unittest.main()
