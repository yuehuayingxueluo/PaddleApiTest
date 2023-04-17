import numpy as np
import paddle
import torch
import unittest
from paddle.fluid.layers.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

def generate_np_inputs_and_dout():
    B_value = 1
    S_value = 40
    H_value = 12

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

class TestFCIncubateCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()

    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"

    def init_threshold(self):
        self.atol = TOLERANCE[self.dtype]["atol"]
        self.rtol = TOLERANCE[self.dtype]["rtol"]

    def init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self.np_input_dir)
        # get np array from npz file
        self.np_x = np_inputs_array["x"]
        self.np_weight = np_inputs_array["weight"]
        self.np_bias = np_inputs_array["bias"]
        self.np_dout = np_inputs_array["dout"]
        #for i in range(3):
            #self.output_size = 1
            #self.output_size = self.np_dout.shape[i] * self.output_size

        self.output_size = self.np_dout.shape[-1]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_x = self.np_x.astype("float16")
            self.np_weight = self.np_weight.astype("float16")
            self.np_bias = self.np_bias.astype("float16")
            self.np_dout = self.np_dout.astype("float16")


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
        bias_static = paddle.static.data(
            'bias',
            shape=self.np_bias.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        bias_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, weight_static, bias_static, dout_static

    def cal_static_res(self, x, weight, bias, dout):
        x_t = x
        weight_t = weight
        bias_t = bias
        dout_t = dout
        if self.dtype == "bfloat16":
            paddle.set_default_dtype("uint16")
            x_t = paddle.cast(x, dtype="uint16")
            weight_t = paddle.cast(weight, dtype="uint16")
            bias_t = paddle.cast(bias, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        else:
            paddle.set_default_dtype(self.dtype)

        out = paddle.fluid.layers.fc(x_t,
                                     size=self.output_size,
                                     num_flatten_dims = self.num_flatten_dims,
                                     param_attr = paddle.fluid.initializer.NumpyArrayInitializer(self.np_weight),
                                     bias_attr= paddle.fluid.initializer.NumpyArrayInitializer(self.np_bias))
        block = paddle.fluid.default_main_program().global_block()
        params = block.all_parameters()
        out_grads = paddle.static.gradients(
            [out], [x] + params, target_gradients=[dout_t]
        )
        paddle.set_default_dtype("float32")
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_static_accuracy(self):
        # get develop static res
        develop_res_array = np.load(self.save_static_res_path)
        out_static_develop = develop_res_array["out_static"]
        out_grads_static_0_develop = develop_res_array["out_grads_static_0"]
        out_grads_static_1_develop = develop_res_array["out_grads_static_1"]
        out_grads_static_2_develop = develop_res_array["out_grads_static_2"]
        out_grads_static_develop = [out_grads_static_0_develop, out_grads_static_1_develop, out_grads_static_2_develop]

        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, weight_static, bias_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    x_static,
                    weight_static,
                    bias_static,
                    dout_static,
            )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(1)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "weight": self.np_weight, "bias": self.np_bias, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]

        # compare incubate static res with develop static res
        np.testing.assert_equal(
            out_static,
            out_static_develop,
            err_msg=(
                'Incubate: compare fluid.layers.fc incubate static forward res with develop static forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_equal(
                out_grads_static[idx],
                out_grads_static_develop[idx],
            err_msg=(
                'Incubate: compare fluid.layers.fc incubate static grad res with develop static grad res failed in %s dtype'
            )
                % self.dtype,
            )


    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, weight_static, bias_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    x_static,
                    weight_static,
                    bias_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(1)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self.np_x, "weight": self.np_weight, "bias": self.np_bias, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self.np_x, "weight": self.np_weight, "bias": self.np_bias, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Incubate: paddle.fluid.layers.fc static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Incubate: paddle.fluid.layers.fc static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )

class TestFCIncubateCase1_FP16(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"

class TestFCIncubateCase1_FP32(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case1_fp32.npz"


class TestFCIncubateCase2_FP32(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.num_flatten_dims = -1
        self.dtype = "float32"
        self.save_static_res_path = "./static_develop_res_case2_fp32.npz"

class TestFCIncubateCase2_FP16(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case2.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case2_fp16.npz"


class TestFCIncubateCase3_FP16(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp16.npz"

class TestFCIncubateCase3_FP32(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case3_fp32.npz"

class TestFCIncubateCase3_BFP16(TestFCIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case3.npz"
        self.num_flatten_dims = -1
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case3_bfp16.npz"
        
if __name__ == '__main__':
    #generate_np_inputs_and_dout()
    print("start run test_linear_incubate.py")
    unittest.main()
    print("finished run test_linear_incubate.py")
