import numpy as np
import paddle
import paddle.distributed as paddle_dist
import base_class
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class TestPaddle(base_class.BaseClass):
    def __init__(self, group, np_input_dir="./inputs_case1.npz", dtype="float32", save_static_res_path="./static_develop_res_case1_float32.npz" , save_eager_res_path="./eager_develop_res_case1_float32.npz", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype, save_static_res_path, save_eager_res_path)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        if paddle.distributed.get_rank() == 0:
            np_inputs_array = np.load(torch_dir)
            self._out_torch = np_inputs_array["torch_out"]
            self._out_grads_torch = np_inputs_array["torch_out_grad"]
        
    
    def _gen_eager_inputs_and_dout(self):
        place = paddle.device.get_device()
        x_eager = paddle.to_tensor(
            self._np_x,
            dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
            place=place,
        )
        x_eager.stop_gradient = False
        dout_eager = paddle.to_tensor(
            self._np_dout,
            dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
            place=place,
        )
        dout_eager.stop_gradient = False
        return x_eager, dout_eager

    def _gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self._np_x.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self._np_dout.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    def _cal_eager_res(self, x, dout):
        dout_t = dout
        if self._dtype == "bfloat16":
            x = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")
        
        x = x.scale(1.0)
        out = paddle_dist.collective._mp_allreduce(x, group=self._group)

        out_grads = paddle.grad(
            [out], [x], grad_outputs=[dout_t]
        )

        out_grads = out_grads[0]     

        if self._dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads = paddle.cast(out_grads, dtype="float32")
        return out, out_grads

    def _cal_static_res(self, x, dout):
        x_t = x
        dout_t = dout
        if self._dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

        out = paddle_dist.collective._mp_allreduce(x, group=self._group)
        out_grads = paddle.static.gradients(
            [out], [x], target_gradients=[dout_t]
        )

        out_grads = out_grads[0]

        if self._dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads =  paddle.cast(out_grads, dtype="float32")
        return out, out_grads

    def _test_eager_accuracy(self):
        x_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager, out_grads_eager = self._cal_eager_res(x_eager, dout_eager)

        del x_eager
        del dout_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()
        out_grads_eager_np = out_grads_eager.numpy()

        del out_eager
        del out_grads_eager
        paddle.device.cuda.empty_cache()

        if paddle.distributed.get_rank() == 0:

            np.savez(self._save_eager_res_path, out_eager=out_eager_np, out_grads_eager=out_grads_eager_np)

            # compare eager res with torch
            try:
                np.testing.assert_allclose(
                    out_eager_np,
                    self._out_torch,
                    self._atol,
                    self._rtol,
                    err_msg=(
                        'Develop: compare mp_allreduce eager forward res with torch failed in %s dtype'
                    )
                    % self._dtype,
                )
            except Exception as e:
                print(e)
                print("eager_accuracy forward {dtype} failed".format(dtype=self._dtype))
            try:
                np.testing.assert_allclose(
                    out_grads_eager_np,
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    err_msg=(
                        'Develop: compare mp_allreduce eager grad res with torch failed in %s dtype'
                    )
                    % self._dtype,
                )
            except Exception as e:
                print(e)
                print("eager_accuracy grad {dtype} failed".format(dtype=self._dtype))
        
    def _test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, dout_static = self._gen_static_inputs_and_dout()
                (out_static, out_grads_static) = self._cal_static_res(
                    x_static,
                    dout_static,
            )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x, "dout": self._np_dout},
                fetch_list=[out_static] + [out_grads_static],
            )
            out_static, out_grads_static = out[0], out[1:]
        
        if paddle.distributed.get_rank() == 0:

            np.savez(self._save_static_res_path, out_static=out_static, out_grads_static=out_grads_static)

            # compare static res with torch
            try:
                np.testing.assert_allclose(
                    out_static,
                    self._out_torch,
                    self._atol,
                    self._rtol,
                    err_msg=(
                        'Develop: compare mp_allreduce static forward res with torch failed in %s dtype'
                    )
                    % self._dtype,
                )
            except Exception as e:
                print(e)
                print("static_accuracy forward {dtype} failed".format(dtype=self._dtype))

            try:
                np.testing.assert_allclose(
                    out_grads_static[0],
                    self._out_grads_torch,
                    self._atol,
                    self._rtol,
                    err_msg=(
                        'Develop: compare mp_allreduce static grad res with torch failed in %s dtype'
                    )
                    % self._dtype,
                )
            except Exception as e:
                print(e)
                print("static_accuracy grad {dtype} failed".format(dtype=self._dtype))

    def _test_eager_stability(self):
        x_eager, dout_eager = self._gen_eager_inputs_and_dout()
        out_eager_baseline, out_grads_eager_baseline = self._cal_eager_res(x_eager, dout_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        out_grads_eager_baseline_np = out_grads_eager_baseline.numpy()
        del out_eager_baseline
        del out_grads_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager, out_grads_eager = self._cal_eager_res(x_eager, dout_eager)
            out_eager = out_eager.numpy()
            out_grads_eager = out_grads_eager.numpy()

            if paddle.distributed.get_rank() == 0:
                try: 
                    np.testing.assert_equal(
                        out_eager,
                        out_eager_baseline_np,
                        err_msg=(
                            'Develop: mp_allreduce eager forward is unstable in %s dtype'
                        )
                        % self._dtype,
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability forward {dtype} failed".format(dtype=self._dtype))
                
                try:
                    np.testing.assert_equal(
                        out_grads_eager,
                        out_grads_eager_baseline_np,
                        err_msg=(
                            'Develop: mp_allreduce eager grad is unstable in %s dtype'
                        )
                        % self._dtype,
                    )
                except Exception as e:
                    print(e)
                    print("eager_stability grad {dtype} failed".format(dtype=self._dtype))

    def _test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static, dout_static = self._gen_static_inputs_and_dout()
                (out_static_pg, out_grads_static_pg) = self._cal_static_res(
                    x_static,
                    dout_static,
                )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x, "dout": self._np_dout},
                fetch_list=[out_static_pg] + [out_grads_static_pg],
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self._np_x, "dout": self._np_dout},
                    fetch_list=[out_static_pg] + [out_grads_static_pg],
                )
                out_static, out_grads_static = out[0], out[1:]

                if paddle.distributed.get_rank() == 0:
                    try:
                        np.testing.assert_equal(
                            out_static,
                            out_static_baseline,
                            err_msg=(
                                'Develop: mp_allreduce static forward is unstable in %s dtype'
                            )
                            % self._dtype,
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability forward {dtype} failed".format(dtype=self._dtype))
                        
                    try: 
                        np.testing.assert_equal(
                            out_grads_static[0],
                            out_grads_static_baseline[0],
                            err_msg=(
                                'Develop: mp_allreduce static grad is unstable in %s dtype'
                            )
                            % self._dtype,
                        )
                    except Exception as e:
                        print(e)
                        print("static_stability forward {dtype} failed".format(dtype=self._dtype))

dtype_list = ["float32", "float16", "bfloat16"]


paddle_dist.init_parallel_env()
world_size = paddle_dist.get_world_size()
group = paddle_dist.new_group([i for i in range(world_size)], backend='nccl')

for case_id in range(2):
    for dtype_id, dtype in enumerate(dtype_list):

        np_input_dir = "./inputs_case{id}.npz".format(id=(case_id + 1))
        save_static_res_path = "./{id}_static_develop_res_case1_{dtype}.npz".format(id=(case_id + 1), dtype=dtype) 
        save_eager_res_path = "./{id}_eager_develop_res_case1_{dtype}.npz".format(id=(case_id + 1), dtype=dtype)
        torch_dir = "{id}_torch_out_{dtype}.npz".format(id=(case_id + 1), dtype=dtype)

        test_paddle = TestPaddle(group, np_input_dir, dtype, save_static_res_path, save_eager_res_path, torch_dir)
        test_paddle._test_eager_accuracy()
        print("eager {dtype} success".format(dtype=dtype))
        test_paddle._test_static_accuracy()
        print("static {dtype} success".format(dtype=dtype))
        test_paddle._test_eager_stability()
        print("eager_stability {dtype} success".format(dtype=dtype))
        test_paddle._test_static_stability()
        print("static_stability {dtype} success".format(dtype=dtype))

        print("{dtype} success".format(dtype=dtype))

