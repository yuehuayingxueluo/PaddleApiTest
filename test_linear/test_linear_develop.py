import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

def generate_np_inputs_and_dout():
    B_value = 1
    S_value = 4096
    H_value = 12288

    x_case1 = np.random.random(size=[S_value , H_value]).astype("float32")
    #x_case1 = np.random.random(size=[B_value, S_value , H_value]).astype("float32")
    weight_case1 = np.random.random(size=[H_value , H_value]).astype("float32")
    dout_case1 = np.random.random(size=[S_value , H_value]).astype("float32")
    bias_case1 = (np.ones( (S_value,1) ) * np.random.random(size=[1 , H_value])).astype("float32")

    x_case2 = np.random.random(size=[S_value , H_value]).astype("float32")
    weight_case2 = np.random.random(size=[H_value , 4 * H_value]).astype("float32")
    dout_case2 = np.random.random(size=[S_value , 4 * H_value]).astype("float32")
    bias_case2 = (np.ones( (S_value,1) ) * np.random.random(size=[1 , 4 * H_value])).astype("float32")

    x_case3 = np.random.random(size=[S_value , 4 * H_value]).astype("float32")
    weight_case3 = np.random.random(size=[4 * H_value ,  H_value]).astype("float32")
    dout_case3 = np.random.random(size=[S_value , H_value]).astype("float32")
    bias_case3 = (np.ones( (S_value,1) ) * np.random.random(size=[1 , H_value])).astype("float32")

    np.savez("./inputs_case1.npz", x = x_case1, weight = weight_case1 , dout = dout_case1 , bias = bias_case1)
    np.savez("./inputs_case2.npz", x = x_case2, weight = weight_case2 , dout = dout_case2 , bias = bias_case2)
    np.savez("./inputs_case3.npz", x = x_case3, weight = weight_case3 , dout = dout_case3 , bias = bias_case3)


class TestLinearDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, weight_torch, dout_torch , bias_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, weight_torch, dout_torch , bias_torch)
        del x_torch
        del weight_torch
        del dout_torch
        del bias_torch
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
                                lambda x: x.cpu().detach().numpy(),
                                out_grads_torch,
                            )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]
        #self.atol = 0.01
        #self.rtol = 0.01

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_weight = np_inputs_array["weight"]
        self.np_dout = np_inputs_array["dout"]
        self.np_bias = np_inputs_array["bias"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_weight = self.np_weight.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
            self.np_bias = self.np_bias.astype("float16")

    def gen_torch_inputs_and_dout(self):
        the_device='cuda:4'
        x_torch = torch.tensor(
            self.np_x,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        weight_torch = torch.tensor(
            self.np_weight,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout_torch = torch.tensor(
            self.np_dout,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        bias_torch = torch.tensor(
            self.np_bias,
            device=the_device,
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, weight_torch, dout_torch , bias_torch

    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        weight_eager = paddle.to_tensor(
            self.np_weight,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        weight_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        bias_eager = paddle.to_tensor(
            self.np_bias,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        bias_eager.stop_gradient = False
        return x_eager, weight_eager, dout_eager , bias_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        weight_static = paddle.static.data(
            'weight',
            shape=self.np_weight.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        weight_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        bias_static = paddle.static.data(
            'bias',
            shape=self.np_bias.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        bias_static.stop_gradient = False
        return x_static, weight_static, dout_static , bias_static

    def cal_torch_res(self, x, weight, dout, bias):
        x_t = x
        weight_t = weight
        dout_t = dout
        bias_t = bias
        if self.dtype == "bfloat16":
            x_t = x.to(dtype=torch.bfloat16)
            weight_t = weight.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
            bias_t = bias.to(dtype=torch.bfloat16)

        weight_t=torch.transpose(weight_t ,dim0=0, dim1=1)
        out = torch.nn.functional.linear(x_t, weight_t , bias_t)
        out_grads = torch.autograd.grad([out], [x, weight, bias], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
        return out, out_grads

    def cal_eager_res(self, x, weight, dout , bias):
        x_t = x
        weight_t = weight
        dout_t = dout
        bias_t = bias
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            weight_t = paddle.cast(weight, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
            bias_t = paddle.cast(bias, dtype="uint16")
        out = paddle.nn.functional.linear(x_t, weight_t, bias_t)
        out_grads = paddle.grad(
            [out], [x, weight , bias], grad_outputs=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def cal_static_res(self, x, weight , dout , bias):
        x_t = x
        weight_t = weight
        dout_t = dout
        bias_t = bias
        if self.dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            weight_t = paddle.cast(weight, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
            bias_t = paddle.cast(bias, dtype="uint16")
        out = paddle.nn.functional.linear(x_t, weight_t, bias_t)
        out_grads = paddle.static.gradients(
            [out], [x, weight, bias], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, weight_eager, dout_eager , bias_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, weight_eager, dout_eager , bias_eager)
        del x_eager
        del weight_eager
        del dout_eager
        del bias_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()
        # save eager res for test_linear_incubate
        np.savez(self.save_eager_res_path, out_eager=out_eager_np, out_grads_eager_0=out_grads_eager_np[0], out_grads_eager_1=out_grads_eager_np[1] , out_grads_eager_2=out_grads_eager_np[2])

        # compare eager res with torch
        np.testing.assert_allclose(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare linear eager forward res with torch failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_eager_np)):
            np.testing.assert_allclose(
                out_grads_eager_np[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare linear eager grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )

    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, weight_static, dout_static , bias_static= self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    weight_static,
                    dout_static,
                    bias_static
            )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "weight": self.np_weight, "dout": self.np_dout , "bias": self.np_bias},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_linear_incubate
        np.savez(self.save_static_res_path, out_static=out_static, out_grads_static_0=out_grads_static[0], out_grads_static_1=out_grads_static[1] , out_grads_static_2=out_grads_static[2])
        #np.savez(self.save_static_res_path, out_static=out_static, out_grads_static_0=out_grads_static[0], out_grads_static_1=out_grads_static[1])

        # compare static res with torch
        np.testing.assert_allclose(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare linear static forward res with torch failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_allclose(
                out_grads_static[idx],
                self.out_grads_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare linear static grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )

    def test_eager_stability(self):
        x_eager, weight_eager, dout_eager , bias_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, weight_eager, dout_eager , bias_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager_baseline,
                            )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, weight_eager, dout_eager , bias_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Develop: paddle.linear eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.linear eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, weight_static, dout_static , bias_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    weight_static,
                    dout_static,
                    bias_static
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "weight": self.np_weight, "dout": self.np_dout , "bias": self.np_bias},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "weight": self.np_weight, "dout": self.np_dout , "bias": self.np_bias},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Develop: paddle.linear static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.linear static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )

class TestLinearDevelopCase1_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase1_FP16")
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"

class TestLinearDevelopCase1_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase1_BFP16")
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"

class TestLinearDevelopCase2_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase2_FP32")
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp32.npz"

class TestLinearDevelopCase2_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase2_FP16")
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_fp16.npz"

class TestLinearDevelopCase3_FP32(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase3_FP32")
        self.np_input_dir = "./inputs_case3.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp32.npz"

class TestLinearDevelopCase3_FP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase3_FP16")
        self.np_input_dir = "./inputs_case3.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_fp16.npz"

class TestLinearDevelopCase2_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase2_BFP16")
        self.np_input_dir = "./inputs_case2.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"

class TestLinearDevelopCase3_BFP16(TestLinearDevelopCase1_FP32):
    def init_params(self):
        print("TestLinearDevelopCase3_BFP16")
        self.np_input_dir = "./inputs_case3.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case3_bfp16.npz"

if __name__ == '__main__':
    generate_np_inputs_and_dout()
    print("finished data generation")
    unittest.main()
    print("finished test_linear_develop.py")
