import numpy as np
import paddle
import torch
import unittest
from paddle.fluid.layers.utils import map_structure
from test_matmul_develop import TestMatmulDevelopCase1_FP32

class TestMatmulIncubateCase1_FP32(TestMatmulDevelopCase1_FP32):
    def test_eager_accuracy(self):
        # get develop eager res
        develop_res_array = np.load(self.save_eager_res_path)
        out_eager_develop = develop_res_array["out_eager"]
        out_eager_grad_0_develop = develop_res_array["out_eager_grad_0"]
        out_eager_grad_1_develop = develop_res_array["out_eager_grad_1"]
        out_eager_grads_develop = [out_eager_grad_0_develop, out_eager_grad_1_develop]

        # calculate incubate eager res
        x_eager, y_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, y_eager, self.transpose_x, self.transpose_y, dout_eager)
        out_eager = out_eager.numpy()
        out_grads_eager = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        
        # compare incubate eager res with develop eager res
        np.testing.assert_equal(
            out_eager,
            out_eager_develop,
            err_msg=(
                'Incubate: compare matmul incubate eager forward res with develop eager forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_eager)):
            np.testing.assert_equal(
                out_grads_eager[idx],
                out_eager_grads_develop,
            err_msg=(
                'Incubate: compare matmul incubate eager grad res with develop eager grad res failed in %s dtype'
            )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        # get develop static res
        develop_res_array = np.load(self.save_static_res_path)
        out_static_develop = develop_res_array["out_static"]
        out_grads_static_0_develop = develop_res_array["out_grads_static_0"]
        out_grads_static_1_develop = develop_res_array["out_grads_static_1"]
        out_grads_static_develop = [out_grads_static_0_develop, out_grads_static_1_develop]

        # calculate incubate static res
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, y_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    y_static,
                    self.transpose_x,
                    self.transpose_y,
                    dout_static,
                )
        exe = paddle.static.Executor(
            place=paddle.CUDAPlace(0)
        )
        exe.run(sp)
        out = exe.run(
            mp,
            feed={"x": self.np_x, "y": self.np_y, "dout": self.np_dout},
            fetch_list=[out_static] + out_grads_static,
        )
        out_static, out_grads_static = out[0], out[1:]
        paddle.disable_static()
        
        # compare incubate static res with develop static res
        np.testing.assert_equal(
            out_static,
            out_static_develop,
            err_msg=(
                'Incubate: compare matmul incubate static forward res with develop static forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_equal(
                out_grads_static[idx],
                out_grads_static_develop,
            err_msg=(
                'Incubate: compare matmul incubate static grad res with develop static grad res failed in %s dtype'
            )
                % self.dtype,
            )

class TestMatmulIncubateCase1_FP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case1.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npy"

class TestMatmulIncubateCase1_BFP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case1.npz"
        self.transpose_x = False
        self.transpose_y = True
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npy"

class TestMatmulIncubateCase2_FP32(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npy"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npy"

class TestMatmulIncubateCase2_FP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npy"

class TestMatmulIncubateCase2_BFP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case2.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npy"

class TestMatmulIncubateCase3_FP32(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case3.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npy"
        self.save_eager_res_path = "./eager_develop_res_case3_fp32.npy"

class TestMatmulIncubateCase3_FP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case3.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case3_fp16.npy"

class TestMatmulIncubateCase3_BFP16(TestMatmulIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./input_case3.npz"
        self.transpose_x = False
        self.transpose_y = False
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npy"
        self.save_eager_res_path = "./eager_develop_res_case3_bfp16.npy"

if __name__ == '__main__':
    unittest.main()
