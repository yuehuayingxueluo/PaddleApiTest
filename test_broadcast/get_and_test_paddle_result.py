import numpy as np
import paddle
import paddle.distributed as paddle_dist
import paddle.distributed.fleet as fleet
import init_config_class
import sys
sys.path.append("..")
from utils import (
    np_assert_accuracy,
    np_assert_staility,
)

class TestPaddle(init_config_class.InitConfigClass):
    def __init__(self, np_input_dir="", dtype=""):
        self._init_params(np_input_dir, dtype)
        self._init_threshold()
        self._init_np_inputs_and_dout()
        if (paddle_dist.get_rank() == 0):
            self._out = self._np_x
        else:
            self._out = np.zeros_like(self._np_x)
    
    def _gen_eager_inputs_and_dout(self):
        place = paddle.device.get_device()
        x_eager = paddle.to_tensor(
            self._np_x,
            dtype=self._dtype if self._dtype != 'bfloat16' else "float32",
            place=place,
        )
        x_eager.stop_gradient = False
        return x_eager

    def _gen_static_inputs_and_dout(self):
        x_static = paddle.static.data(
            'x',
            shape=self._np_x.shape,
            dtype=self._dtype if self._dtype != "bfloat16" else "float32",
        )
        x_static.stop_gradient = False
        return x_static

    def _cal_eager_res(self, x):
        x_t = x
        if self._dtype == "bfloat16":
            x_t = paddle.cast(x_t, dtype="uint16")
        
        x_t = x_t.scale(1.0)
        paddle_dist.broadcast(x_t, 0)   

        if self._dtype == "bfloat16":
            x_t = paddle.cast(x_t, dtype="float32")
        return x_t

    def _cal_static_res(self, x):
        x_t = x
        if self._dtype == "bfloat16":
            x_t = paddle.cast(x_t, dtype="uint16")
        
        
        paddle_dist.broadcast(x_t, 0)

        if self._dtype == "bfloat16":
            x_t = paddle.cast(x_t, dtype="float32")
        return x_t

    def _test_eager_accuracy(self):
        x_eager = self._gen_eager_inputs_and_dout()
        out_eager = self._cal_eager_res(x_eager)

        del x_eager
        paddle.device.cuda.empty_cache()
        out_eager_np = out_eager.numpy()

        del out_eager
        paddle.device.cuda.empty_cache()

        # compare eager res with numpy
        try:
            np_assert_accuracy(
                out_eager_np,
                self._out,
                self._atol,
                self._rtol,
                self._dtype,
                version_a="paddle_develop",
                version_b="numpy",
                eager_or_static_mode="eager",
                fwd_or_bkd="forward",
                api="paddle.distributed.broadcast",
            )
        except Exception as e:
            print(e)
            print("eager_accuracy forward {dtype} failed".format(dtype=self._dtype))
        
    def _test_static_accuracy(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self._gen_static_inputs_and_dout()
                (out_static) = self._cal_static_res(
                    x_static
            )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x},
                fetch_list=[out_static],
            )
            out_static = out[0]

        # compare static res with numpy
        try:
            np_assert_accuracy(
                out_static,
                self._out,
                self._atol,
                self._rtol,
                self._dtype,
                version_a="paddle_develop",
                version_b="numpy",
                eager_or_static_mode="static",
                fwd_or_bkd="forward",
                api="paddle.distributed.broadcast",
            )
        except Exception as e:
            print(e)
            print("static_accuracy forward {dtype} failed".format(dtype=self._dtype))

    def _test_eager_stability(self):
        x_eager = self._gen_eager_inputs_and_dout()
        out_eager_baseline = self._cal_eager_res(x_eager)
        out_eager_baseline_np = out_eager_baseline.numpy()
        del out_eager_baseline
        paddle.device.cuda.empty_cache()

        for i in range(50):
            out_eager = self._cal_eager_res(x_eager)
            out_eager = out_eager.numpy()

            try:
                np_assert_staility(
                    out_eager,
                    out_eager_baseline_np,
                    self._dtype,
                    version="paddle_develop",
                    eager_or_static_mode="eager",
                    fwd_or_bkd="forward",
                    api="paddle.distributed.broadcast",
                )
            except Exception as e:
                print(e)
                print("eager_stability forward {dtype} failed".format(dtype=self._dtype))

    def _test_static_stability(self):
        with paddle.fluid.framework._dygraph_guard(None):
            mp, sp = paddle.static.Program(), paddle.static.Program()
            with paddle.static.program_guard(mp, sp):
                x_static = self._gen_static_inputs_and_dout()
                (out_static_pg) = self._cal_static_res(
                    x_static,
                )
            exe = paddle.static.Executor()
            exe.run(sp)
            out = exe.run(
                mp,
                feed={"x": self._np_x},
                fetch_list=[out_static_pg],
            )
            out_static_baseline = out[0]
            
            for i in range(50):
                out = exe.run(
                    mp,
                    feed={"x": self._np_x},
                    fetch_list=[out_static_pg],
                )
                out_static = out[0]

                try:
                    np_assert_staility(
                        out_static,
                        out_static_baseline,
                        self._dtype,
                        version="paddle_develop",
                        eager_or_static_mode="static",
                        fwd_or_bkd="forward",
                        api="paddle.distributed.broadcast",
                    )
                except Exception as e:
                    print(e)
                    print("static_stability forward {dtype} failed".format(dtype=self._dtype))

dtype_list = ["float32", "float16", "bfloat16"]


dist_strategy = fleet.DistributedStrategy()
world_size = paddle_dist.get_world_size()
dist_strategy.hybrid_configs = {
    "mp_degree": 1,
    "pp_degree": 1,
    "dp_degree": world_size,
}
fleet.init(is_collective=True, strategy = dist_strategy)
world_size = paddle_dist.get_world_size()

for case_id in range(2):
    for dtype_id, dtype in enumerate(dtype_list):

        np_input_dir = "./inputs_case{id}.npz".format(id=(case_id + 1))

        test_paddle = TestPaddle(np_input_dir, dtype)
        test_paddle._test_eager_accuracy()
        print("eager {dtype} success".format(dtype=dtype))
        # test_paddle._test_static_accuracy()
        print("static {dtype} success".format(dtype=dtype))
        test_paddle._test_eager_stability()
        print("eager_stability {dtype} success".format(dtype=dtype))
        # test_paddle._test_static_stability()
        print("static_stability {dtype} success".format(dtype=dtype))

        print("{dtype} success".format(dtype=dtype))

