import numpy as np
import paddle
import torch
import unittest
from paddle.fluid.layers.utils import map_structure
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestCrossEntropyLossIncubateCase1_FP32(unittest.TestCase):
    def setUp(self):
        self.init_params()
        self.init_threshold()
        self.init_np_inputs_and_dout()

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
        self.np_logits = np_inputs_array["logits"][0]
        self.np_label = np_inputs_array["label"]
        self.np_dout = np_inputs_array["dout"]
        # convert np array dtype
        if self.dtype == "float16":
            self.np_logits = self.np_logits.astype("float16")
            self.np_label = self.np_label.astype("int64")
            self.np_dout = self.np_dout.astype("float16")
    
    def gen_eager_inputs_and_dout(self):
        logits_eager = paddle.to_tensor(
            self.np_logits,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        logits_eager.stop_gradient = False
        lablel_eager = paddle.to_tensor(
            self.np_label,
            dtype="int64",
            place="gpu",
        )
        dout_eager = paddle.to_tensor(
            self.np_dout,
            dtype=self.dtype if self.dtype != 'bfloat16' else "float32",
            place="gpu",
        )
        return logits_eager, lablel_eager, dout_eager

    def gen_static_inputs_and_dout(self):
        logits_static = paddle.static.data(
            'logits',
            shape=self.np_logits.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        logits_static.stop_gradient = False
        label_static = paddle.static.data(
            'label',
            shape=self.np_label.shape,
            dtype="int64",
        )
        # label_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self.np_dout.shape,
            dtype=self.dtype if self.dtype != "bfloat16" else "float32",
        )
        return logits_static, label_static, dout_static

    def cal_eager_res(self, logits, label, dout):
        logits_t = logits
        label_t = label
        dout_t = dout
        if self.dtype == "bfloat16":
            logits_t = paddle.cast(logits, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.nn.functional.softmax_with_cross_entropy(
            logits_t,
            label_t,
            ignore_index=-100,
            soft_label=False,
            axis=-1,
        )
        out_grads = paddle.grad(
            [out], [logits_t], grad_outputs=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads
    def cal_static_res(self, logits, label, dout):
        logits_t = logits
        label_t = label
        dout_t = dout
        if self.dtype == "bfloat16":
            logits_t = paddle.cast(logits, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        out = paddle.nn.functional.softmax_with_cross_entropy(
            logits_t,
            label_t,
            ignore_index=-100,
            soft_label=False,
            axis=-1,
        )
        out_grads = paddle.static.gradients(
            [out], [logits_t], target_gradients=[dout_t]
        )
        if self.dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
        return out, out_grads

    def test_eager_accuracy(self):

        # get develop eager res
        develop_res_array = np.load(self.save_eager_res_path)
        out_eager_develop = develop_res_array["out_eager"]
        out_eager_grad_0_develop = develop_res_array["out_grads_eager_0"][0]
        out_eager_grads_develop = [out_eager_grad_0_develop]

        logits_eager, label_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(logits_eager, label_eager, dout_eager)
        del logits_eager
        del label_eager 
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
        # compare incubate eager res with develop eager res
        np.testing.assert_equal(
            out_eager_np,
            out_eager_develop,
            err_msg=(
                'Incubate: compare cross_entropy incubate eager forward res with develop eager forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_eager_np)):
            np.testing.assert_equal(
                out_grads_eager_np[idx],
                out_eager_grads_develop[idx],
            err_msg=(
                'Incubate: compare cross_entropy incubate eager grad res with develop eager grad res failed in %s dtype'
            )
                % self.dtype,
            )
    
    def test_static_accuracy(self):
        # get develop static res
        develop_res_array = np.load(self.save_static_res_path)
        out_static_develop = develop_res_array["out_static"]
        out_grads_static_0_develop = develop_res_array["out_grads_static_0"][0]
        out_grads_static_develop = [out_grads_static_0_develop]

        # calculate incubate static res
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                logits_static, label_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self.cal_static_res(
                    logits_static,
                    label_static,
                    dout_static,
            )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"logits": self.np_logits, "label": self.np_label, "dout": self.np_dout},
                fetch_list=[out_static] + out_grads_static,
            )
            out_static, out_grads_static = out[0], out[1:]
        
        # compare incubate static res with develop static res
        np.testing.assert_equal(
            out_static,
            out_static_develop,
            err_msg=(
                'Incubate: compare cross_entropy incubate static forward res with develop static forward res failed in %s dtype'
            )
            % self.dtype,
        )
        for idx in range(len(out_grads_static)):
            np.testing.assert_equal(
                out_grads_static[idx],
                out_grads_static_develop[idx],
            err_msg=(
                'Incubate: compare cross_entropy incubate static grad res with develop static grad res failed in %s dtype'
            )
                % self.dtype,
            )

    def test_eager_stability(self):
        logits_eager, label_eager, dout_eager = self.gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self.cal_eager_res(logits_eager, label_eager, dout_eager)
        out_eager_baseline_np = out_eager.numpy()
        out_grads_eager_baseline_np = map_structure(
                                lambda x: x.numpy(),
                                out_grads_eager,
                            )
        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self.cal_eager_res(logits_eager, label_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = map_structure(
                                    lambda x: x.numpy(),
                                    out_grads_eager,
                                )
            np.testing.assert_equal(
                out_eager,
                out_eager_baseline_np,
                err_msg=(
                    'Incubate: paddle.nn.functional.softmax_with_cross_entropy eager forward is unstable in %s dtype'
                )
                % self.dtype,
            )
            for idx in range(len(out_grads_eager)):
                np.testing.assert_equal(
                    out_grads_eager[idx],
                    out_grads_eager_baseline_np[idx],
                    err_msg=(
                        'Incubate: paddle.nn.functional.softmax_with_cross_entropy eager grad is unstable in %s dtype'
                    )
                    % self.dtype,
                )

    def test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                logit_static, label_static, dout_static = self.gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self.cal_static_res(
                    logit_static,
                    label_static,
                    dout_static,
                )
            exe = paddle.static.Executor(
                place=paddle.CUDAPlace(0)
            )
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"logits": self.np_logits, "label": self.np_label, "dout": self.np_dout},
                fetch_list=[out_static_pg] + out_grads_static_pg,
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"logits": self.np_logits, "label": self.np_label, "dout": self.np_dout},
                    fetch_list=[out_static_pg] + out_grads_static_pg,
                )
                out_static, out_grads_static = out[0], out[1:]
                np.testing.assert_equal(
                    out_static,
                    out_static_baseline,
                    err_msg=(
                        'Incubate: paddle.nn.functional.softmax_with_cross_entropy static forward is unstable in %s dtype'
                    )
                    % self.dtype,
                )
                for idx in range(len(out_grads_static)):
                    np.testing.assert_equal(
                        out_grads_static[idx],
                        out_grads_static_baseline[idx],
                        err_msg=(
                            'Incubate: paddle.nn.functional.softmax_with_cross_entropy static grad is unstable in %s dtype'
                        )
                        % self.dtype,
                    )


class TestCrossEntropyDevelopCase1_FP16(TestCrossEntropyLossIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "float16"
        self.save_static_res_path = "./static_develop_res_case1_fp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case1_fp16.npz"

class TestCrossEntropyDevelopCase2_BFP16(TestCrossEntropyLossIncubateCase1_FP32):
    def init_params(self):
        self.np_input_dir = "./inputs_case1.npz"
        self.dtype = "bfloat16"
        self.save_static_res_path = "./static_develop_res_case2_bfp16.npz"
        self.save_eager_res_path = "./eager_develop_res_case2_bfp16.npz"

if __name__ == '__main__':
    unittest.main()
