import numpy as np
import paddle
import torch
import unittest
from paddle.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

def generate_np_inputs_and_dout():

    x_case1 = np.random.randint(1,3072,size=[1,4096])
    w_case1 = np.random.random(size=[3072, 12288]).astype("float32")
    dout_case1 = np.random.random(size=[1,4096, 12288]).astype("float32")
    np.savez("./inputs_case1.npz", x = x_case1, weight = w_case1, dout = dout_case1)

class TestEmbeddingDevelopCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()
        x_torch, w_torch, dout_torch = self.gen_torch_inputs_and_dout()
        out_torch, out_grads_torch = self.cal_torch_res(x_torch, w_torch, dout_torch)
        del x_torch 
        del w_torch 
        del dout_torch 
        self.out_torch = out_torch.cpu().detach().numpy()
        self.out_grads_torch = map_structure(
                                lambda x: x.cpu().numpy(),
                                out_grads_torch,
                            )
        del out_torch, out_grads_torch
        torch.cuda.empty_cache()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.padding_idx = None
        self.sparse  =False
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
        self.np_w = np_inputs_array["weight"]
        self.np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_w = self.np_w.astype("float16")
            self.np_dout = self.np_dout.astype("float16")
    
    def gen_torch_inputs_and_dout(self):
        x_torch = torch.tensor(
            self.np_x,
            device='cuda',
        )
        w_torch = torch.tensor(
            self.np_w,
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
        return x_torch, w_torch, dout_torch
    
    def gen_eager_inputs_and_dout(self):
        x_eager = paddle.to_tensor(
            self.np_x,
            place="gpu",
            dtype=self.np_x.dtype,
        )
        x_eager.stop_gradient = True
        w_eager = paddle.to_tensor(
            self.np_w,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        w_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        dout_eager.stop_gradient = False
        return x_eager, w_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self.np_x.shape,
            dtype=self.np_x.dtype,
        )
        x_static.stop_gradient = True
        w_static = paddle.static.data(
            'w',
            shape=self.np_w.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        w_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, w_static, dout_static

    def cal_torch_res(self, x, w, dout):
        x_t = x
        w_t = w
        dout_t = dout
        if self.dtype == "bfloat16":
            w_t = w.to(dtype=torch.bfloat16)
            dout_t = dout.to(dtype=torch.bfloat16)
        out = torch.nn.functional.embedding(x_t, w_t)
        out_grads = torch.autograd.grad([out], [w], grad_outputs=[dout_t])
        if self.dtype == "bfloat16":
            out = out.to(dtype=torch.float32)
            out_grads = map_structure(lambda x: x.to(dtype=torch.float32), out_grads)
        return out, out_grads

    def cal_eager_res(self, x, w, dout):
        x_t = x
        w_t = w
        dout_t = dout
        if self.dtype == "bfloat16":
            w_t = paddle.cast(w, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        embedding = paddle.nn.Embedding(self.np_w.shape[0], self.np_w.shape[1])
        paddle.assign(w_t, embedding.weight)
        out = embedding(x_t)
        out_grads = paddle.grad(
            [out], [w], grad_outputs=[dout_t],
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def cal_static_res(self, x, w, dout):
        x_t = x
        w_t = w
        dout_t = dout
        if self.dtype == "bfloat16":
            w_t = paddle.cast(w, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        embedding = paddle.nn.Embedding(self.np_w.shape[0], self.np_w.shape[1])
        paddle.assign(w_t, embedding.weight)
        out = embedding(x_t)
        out_grads = paddle.static.gradients(
            [out], [w], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = map_structure(lambda x: paddle.cast(x, dtype='float32'), out_grads)
        return out, out_grads

    def test_eager_accuracy(self):
        # calculate develop eager res
        x_eager, w_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(x_eager, w_eager, dout_eager)
        del x_eager
        del w_eager 
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
        np.savez(self.save_eager_res_path, out_eager=out_eager_np, out_grads_eager_0=out_grads_eager_np[0])
        # compare eager res with torch
        np.testing.assert_allclose(
            out_eager_np,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare embedding eager forward res with torch failed in %s dtype'
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
                    'Develop: compare embedding eager grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, w_static, dout_static = self.gen_static_inputs_and_dout()
                out_static, out_grads_static = self.cal_static_res(x_static, w_static, dout_static)
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "w": self.np_w, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # save static res for test_scale_incubate
        np.savez(self.save_static_res_path, out_static=out_static, out_grads_static_0=out_grads_static[0])
        
        # compare static res with torch
        np.testing.assert_allclose(
            out_static,
            self.out_torch,
            self.atol,
            self.rtol,
            err_msg=(
                'Develop: compare embedding static forward res with torch failed in %s dtype'
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
                    'Develop: compare embedding static grad res with torch failed in %s dtype'
                )
                % self.dtype,
            )

    def test_eager_stability(self):
        x_eager, w_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self.cal_eager_res(x_eager, w_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager_baseline,
                            )
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(x_eager, w_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Develop: paddle.nn.Embedding eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Develop: paddle.nn.Embedding eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, w_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    w_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "w": self.np_w, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "w": self.np_w, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Develop: paddle.nn.Embedding static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Develop: paddle.nn.Embedding static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )



class TestEmbeddingDevelopCase1_FP16(TestEmbeddingDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"

class TestEmbeddingDevelopCase1_BFP16(TestEmbeddingDevelopCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case1_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_bfp16.npz"

if __name__ == '__main__':
    generate_np_inputs_and_dout()
    unittest.main()

