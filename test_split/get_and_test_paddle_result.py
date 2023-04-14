import numpy as np
import paddle
import paddle.distributed as paddle_dist
import paddle.distributed.fleet as fleet
import base_class
import random
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    fleet.meta_parallel.model_parallel_random_seed(seed)

class TestPaddle(base_class.BaseClass):
    def __init__(self, group, np_input_dir="./inputs_case1.npz", dtype="float32", save_static_res_path="./static_develop_res_case1_float32.npz" , save_eager_res_path="./eager_develop_res_case1_float32.npz", torch_dir="1_torch_out_float32.npz"):
        self._init_params(np_input_dir, dtype, save_static_res_path, save_eager_res_path)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        self._group = group
        world_size = paddle.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
        np_inputs_array = np.load(torch_dir)
        self._out_torch = np_inputs_array["torch_out"]

        if(self._axis == 0):
            self._x = np.array_split(self._np_x, world_size, -1)[rank]
            self._weight = np.array_split(self._np_weight, world_size, 0)[rank]
            self._bias = self._np_bias
            self._out_grads_torch  = np.array_split(np_inputs_array["torch_out_grad"][0], world_size, -1)[rank]
            # out_grads_torch_tmp[1] = np.array_split(np_inputs_array["torch_out_grad"][1], world_size, 0)[rank]
            # out_grads_torch_tmp[0] = np_inputs_array["torch_out_grad"][2]
        else:
            self._x = self._np_x
            self._weight = np.array_split(self._np_weight, world_size, 1)[rank]
            self._bias = np.array_split(self._np_bias, world_size, 0)[rank]
            self._out_grads_torch = np_inputs_array["torch_out_grad"][0]
            # out_grads_torch_tmp[1] = np.array_split(np_inputs_array["torch_out_grad"][1], world_size, 1)[rank]
            # out_grads_torch_tmp[0] = np.array_split(np_inputs_array["torch_out_grad"][2], world_size, 0)[rank]
    
    # def _gen_eager_inputs_and_dout(self):
    #     place = paddle.device.get_device()
    #     x_eager = paddle.to_tensor(
    #         self._x,
    #         dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
    #         place=place,
    #     )
    #     x_eager.stop_gradient = False
    #     weight_eager = paddle.to_tensor(
    #         self._weight,
    #         dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
    #         place=place,
    #     )
    #     weight_eager.stop_gradient = False
    #     bias_eager = paddle.to_tensor(
    #         self._bias,
    #         dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
    #         place=place,
    #     )
    #     bias_eager.stop_gradient = False
    #     dout_eager = paddle.to_tensor(
    #         self._np_dout,
    #         dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
    #         place=place,
    #     )
    #     dout_eager.stop_gradient = False
    #     return x_eager, weight_eager, bias_eager,dout_eager

    def _gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self._x.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        # weight_static = paddle.static.data(
        #     'weight',
        #     shape=self._weight.shape,
        #     dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        # )
        # weight_static.stop_gradient = False
        # bias_static = paddle.static.data(
        #     'bias',
        #     shape=self._bias.shape,
        #     dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        # )
        # bias_static.stop_gradient = False
        dout_static = paddle.static.data(
            'dout',
            shape=self._np_dout.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        dout_static.stop_gradient = False
        return x_static, dout_static

    # def _cal_eager_res(self, x, dout):
    #     x_t = x
    #     # weight_t = weight
    #     # bias_t = bias
    #     dout_t = dout
    #     if self._dtype == "bfloat16":
    #         x_t = paddle.cast(x, dtype="uint16")
    #         # weight_t = paddle.cast(weight, dtype="uint16")
    #         # bias_t = paddle.cast(bias, dtype="uint16")
    #         dout_t = paddle.cast(dout, dtype="uint16")
        
    #     out = paddle.distributed.split(
    #             x=x_t,
    #             size=(self._np_weight.shape[0],
    #             self._np_weight.shape[1]), 
    #             operation="linear", 
    #             axis=self._axis, num_partitions=2, 
    #             weight_attr=paddle.nn.initializer.NumpyArrayInitializer(self._weight),
    #             bias_attr=paddle.nn.initializer.NumpyArrayInitializer(self._bias),
    #             mp_group=self._group)

    #     out_grads = paddle.grad(
    #         [out], [x_t], grad_outputs=[dout_t]
    #     )

    #     if self._dtype == "bfloat16":
    #         out = paddle.cast(out, dtype="float32")
    #         out_grads = paddle.cast(out_grads[0], dtype="float32")

    #     return out, out_grads[0]

    def _cal_static_res(self, x, dout):
        x_t = x
        dout_t = dout

        if self._dtype == "bfloat16":
            x_t = paddle.cast(x, dtype="uint16")
            dout_t = paddle.cast(dout, dtype="uint16")

        origin_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype(self._dtype)

        out = paddle.distributed.split(
                x=x_t,
                size=(self._np_weight.shape[0],
                self._np_weight.shape[1]), 
                operation="linear", 
                axis=self._axis,
                num_partitions=2, 
                weight_attr=paddle.nn.initializer.NumpyArrayInitializer(self._weight),
                bias_attr=paddle.nn.initializer.NumpyArrayInitializer(self._bias))
        
        paddle.set_default_dtype(origin_dtype)

        out_grads = paddle.static.gradients(
            [out], [x_t], target_gradients=[dout_t]
        )

        if self._dtype == "bfloat16":
            out = paddle.cast(out, dtype="float32")
            out_grads =  paddle.cast(out_grads[0], dtype="float32")
        return out, out_grads
        
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
                feed={"x": self._x, "dout": self._np_dout},
                fetch_list=[out_static] + [out_grads_static],
            )
            out_static, out_grads_static = out[0], out[1:]
            out_grads_static = out_grads_static[0]

        np.savez(self._save_static_res_path, out_static=out_static, out_grads_static=out_grads_static)

        # compare static res with torch
        try:
            np.testing.assert_allclose(
                out_static,
                self._out_torch,
                self._atol,
                self._rtol,
                err_msg=(
                    'Develop: compare split static forward res with torch failed in %s dtype'
                )
                % self._dtype,
            )
        except Exception as e:
            print(e)
            print("static_accuracy forward {dtype} failed".format(dtype=self._dtype))
        try:
            np.testing.assert_allclose(
                out_grads_static,
                self._out_grads_torch,
                self._atol,
                self._rtol,
                err_msg=(
                    'Develop: compare split static grad res with torch failed in %s dtype'
                )
                % self._dtype,
            )
        except Exception as e:
            print(e)
            print("static_accuracy grad {dtype} failed".format(dtype=self._dtype))

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
                feed={"x": self._x, "dout": self._np_dout},
                fetch_list=[out_static_pg] + [out_grads_static_pg],
            )
            out_static_baseline, out_grads_static_baseline = out[0], out[1:]
            
            for i in range(1):
                out = exe.run(
                    mp,
                    feed={"x": self._x, "dout": self._np_dout},
                    fetch_list=[out_static_pg] + [out_grads_static_pg],
                )
                out_static, out_grads_static = out[0], out[1:]

                try:
                    np.testing.assert_equal(
                        out_static,
                        out_static_baseline,
                        err_msg=(
                            'Develop: split static forward is unstable in %s dtype'
                        )
                        % self._dtype,
                    )
                except Exception as e:
                    print(e)
                    print("static_stability forward {dtype} failed".format(dtype=self._dtype))
                try: 
                    np.testing.assert_equal(
                        out_grads_static,
                        out_grads_static_baseline,
                        err_msg=(
                            'Develop: split static grad is unstable in %s dtype'
                        )
                        % self._dtype,
                    )
                except Exception as e:
                    print(e)
                    print("static_stability grad {dtype} failed".format(dtype=self._dtype))

dist_strategy = fleet.DistributedStrategy()
world_size = paddle_dist.get_world_size()
dist_strategy.hybrid_configs = {
    "mp_degree": world_size,
    "pp_degree": 1,
    "dp_degree": 1,
}
paddle_dist.fleet.init(is_collective=True, strategy = dist_strategy)

set_random_seed(1024)

group = paddle_dist.new_group([i for i in range(world_size)], backend='nccl')

case_num = 5
rank = paddle_dist.get_rank()

dtype_list = ["float32", "float16", "bfloat16"]

for case_id in range(case_num):
    for dtype_id, dtype in enumerate(dtype_list):

        np_input_dir = "./inputs_case{id_b}.npz".format(id_b=(case_id + 1))
        save_static_res_path = "./{id}_static_develop_res_case1_{dtype}_{rank}.npz".format(id=case_id+1, dtype=dtype, rank=rank)
        save_eager_res_path = "./{id}_eager_develop_res_case1_{dtype}_{rank}.npz".format(id=case_id+1, dtype=dtype, rank=rank)
        torch_dir = "./{id}_torch_out_{dtype}.npz".format(id=case_id+1, dtype=dtype)

        test_paddle = TestPaddle(group, np_input_dir, dtype, save_static_res_path, save_eager_res_path, torch_dir)
        print("case {id} {dtype} static start".format(id=case_id + 1, dtype=dtype))
        test_paddle._test_static_accuracy()
        print("case {id} {dtype} static finish".format(id=case_id + 1, dtype=dtype))
        print("case {id} {dtype} static_stability start".format(id=case_id + 1, dtype=dtype))
        test_paddle._test_static_stability()
        print("case {id} {dtype} static_stability finish".format(id=case_id + 1, dtype=dtype))

        print("case {id} {dtype} finish".format(id=case_id + 1, dtype=dtype))

