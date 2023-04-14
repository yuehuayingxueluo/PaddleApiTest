import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type


def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, 4096, 12288]).astype("float32")
    dim_case1 = -1
    num_case1 = 2
    dout1_case1 = np.random.random(size=[1, 4096, 6144]).astype("float32")
    dout2_case1 = np.random.random(size=[1, 4096, 6144]).astype("float32")

    np.savez("./inputs_case1.npz", x=x_case1, dim=dim_case1, num=num_case1, dout1 = dout1_case1, dout2 = dout2_case1)

class TestSplitDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, dim_torch, num_torch, dout1_torch, dout2_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, dim_torch, num_torch, dout1_torch, dout2_torch)
        del x_torch 
        del dim_torch
        del num_torch
        del dout1_torch
        del dout2_torch
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

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp32.npz"
    
    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_dim = int(np_inputs_array["dim"])
        self.np_num = int(np_inputs_array["num"])
        self.np_dout1 = np_inputs_array["dout1"]
        self.np_dout2 = np_inputs_array["dout2"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_dout1 = self.np_dout1.astype("float16")
            self.np_dout2 = self.np_dout2.astype("float16")
    
    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dim_torch = self.np_dim
        num_torch = 6144
        dout1_torch = torch.tensor(
            self.np_dout1,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        dout2_torch = torch.tensor(
            self.np_dout2,
            device='cuda',
            dtype=convert_dtype_to_torch_type(self.dtype)
            if self.dtype != 'bfloat16'
            else torch.float32,
            requires_grad=True,
        )
        return x_torch, dim_torch, num_torch, dout1_torch, dout2_torch
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        x_eager.stop_gradient = False
        dim_eager = self.np_dim
        num_eager = self.np_num
        dout1_eager = paddle.to_tensor(
            self.np_dout1,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout1_eager.stop_gradient = False
        dout2_eager = paddle.to_tensor(
            self.np_dout2,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout2_eager.stop_gradient = False
        return x_eager, dim_eager, num_eager, dout1_eager, dout2_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dim_static = self.np_dim
        num_static = self.np_num
        dout1_static = paddle.static.data(
            'dout1',
            shape=self.np_dout1.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout1_static.stop_gradient = False
        dout2_static = paddle.static.data(
            'dout2',
            shape=self.np_dout2.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout2_static.stop_gradient = False
        return x_static, dim_static, num_static, dout1_static, dout2_static

    def cal_torch_res(self, x, dim, num, dout1, dout2):
        if self.dtype == "bfloat16":
            x = x.to(dtype=torch.bfloat16)
            dout1 = dout1.to(dtype=torch.bfloat16)
            dout2 = dout2.to(dtype=torch.bfloat16)
        out = torch.split(x, num, dim)
        out_grads = torch.autograd.grad(out, [x], grad_outputs=[dout1, dout2])
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: x.to(dtype=torch.float32), out)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, dim, num, dout1, dout2):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout1 = paddle.cast(dout1, dtype="uint16")
            dout2 = paddle.cast(dout2, dtype="uint16")
        out = paddle.split(x, num, dim)
        out_grads = paddle.grad(
            out, [x], grad_outputs=[dout1, dout2]
        )
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: paddle.cast(x, dtype='float32'), out)
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def cal_static_res(self, x, dim, num, dout1, dout2):
        if self.dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout1 = paddle.cast(dout1, dtype="uint16")
            dout2 = paddle.cast(dout2, dtype="uint16")
        out = paddle.split(x, num, dim)
        out_grads = paddle.static.gradients(
            out, [x], target_gradients=[dout1, dout2]
        )
        if self.dtype == "bfloat16":
            out = map_structure(lambda x: paddle.cast(x, dtype='float32'), out)
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        x_eager, dim_eager, num_eager, dout1_eager, dout2_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, dim_eager, num_eager, dout1_eager, dout2_eager)
        del x_eager
        del dim_eager
        del num_eager
        del dout1_eager
        del dout2_eager
        paddle.device.cuda.empty_cache()
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
        paddle.device.cuda.empty_cache()
        # save eager res for test_scale_incubate
        np.savez(
            self.save_eager_res_path,
            out_eager_0=out_eager_np[0],
            out_eager_1=out_eager_np[1],
            out_grads_eager_0=out_grads_eager_np[0],
            )
        
        # compare eager res with torch
        for idx in range(len(out_eager_np)):
            np.testing.assert_allclose(
                out_eager_np[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare split eager forward res with torch failed in %s dtype'
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
                    'Develop: compare split eager grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, dim_static, num_static, dout1_static, dout2_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    dim_static,
                    num_static,
                    dout1_static,
                    dout2_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout1": self.np_dout1, "dout2": self.np_dout2},
                fetch_list=out_static + out_grads_static,
            )
            out_static, out_grads_static = out[:2], out[2:]

        # save static res for test_scale_incubate
        np.savez(
            self.save_static_res_path,
            out_static_0=out_static[0],
            out_static_1=out_static[1],
            out_grads_static_0=out_grads_static[0],
            )
        
        # compare static res with torch
        for idx in range(len(out_static)):
            np.testing.assert_allclose(
                out_static[idx],
                self.out_torch[idx],
                self.atol,
                self.rtol,
                err_msg=(
                    'Develop: compare split static forward res with torch failed in %s dtype'
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
                    'Develop: compare split static grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )

    def test_eager_stability(self):
        x_eager, dim_eager, num_eager, dout1_eager, dout2_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, dim_eager, num_eager, dout1_eager, dout2_eager)
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
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, dim_eager, num_eager, dout1_eager, dout2_eager)
            out_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_eager,
                                    )
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                    )
            for idx in range(len(out_eager)):
                np.testing.assert_equal(
                    out_eager[idx],
                    out_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.split eager forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.split eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, dim_static, num_static, dout1_static, dout2_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    dim_static,
                    num_static,
                    dout1_static,
                    dout2_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "dout1": self.np_dout1, "dout2": self.np_dout2},
                fetch_list=out_static_pg + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[:2], out[2:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "dout1": self.np_dout1, "dout2": self.np_dout2},
                    fetch_list=out_static_pg + out_grads_static_pg,
                )
                out_static, out_grads_static = out[:2], out[2:]
                for idx in range(len(out_static)):
                    np.testing.assert_equal(
                        out_static[idx],
                        out_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.split static forward is unstable in %s dtype'
                        )
                        % self.dtype,
                    )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.split static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )


class TestSplitDevelopCase1_FP16(TestSplitDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"


class TestSplitDevelopCase1_BFP16(TestSplitDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"


if __name__ == '__main__':
    generate_np_inputs_and_dout()
    unittest.main()
