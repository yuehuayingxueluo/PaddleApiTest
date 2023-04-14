import paddle
from common import io
from common import framework
import numpy as np
import attribute

class Cast(framework.Framework):
  @staticmethod
  def launch_static(input_data, attr):
    with paddle.fluid.framework._dygraph_guard(None):
      mp, sp = paddle.static.Program(), paddle.static.Program()
      with paddle.static.program_guard(mp, sp):
        x_static_fp32 = paddle.static.data(
          'x',
          shape=input_data.x.shape,
          dtype="float32",
        )
        t_static_fp32 = paddle.static.data(
          'out_t',
          shape=input_data.out_t.shape,
          dtype="float32",
        )
        x_static = paddle.cast(x_static_fp32, dtype=attr.src_dtype)
        t_static = paddle.cast(t_static_fp32, dtype=attr.tgt_dtype)
        x_static.stop_gradient = False
        out_static = paddle.cast(x_static, dtype=attr.tgt_dtype)
        out_grads_static = paddle.static.gradients(
          [out_static], [x_static], [t_static])
        out_static_fp32 = paddle.cast(out_static, "float32")
        out_static_grads_fp32 = paddle.cast(out_grads_static[0], "float32")
        exe = paddle.static.Executor(
          place=paddle.CUDAPlace(0)
        )
        exe.run(sp)
        ret = exe.run(
          mp,
          feed={"x": input_data.x, "out_t": input_data.out_t},
          fetch_list=[out_static_fp32, out_static_grads_fp32],
        )
        return ret

  @staticmethod
  def launch_eager(input_data, attr):
    input_tensor_fp32 = paddle.to_tensor(
      input_data.x,
      dtype="float32",
      place="gpu",
    )
    t_tensor_fp32 = paddle.to_tensor(
      input_data.out_t,
      dtype="float32",
      place="gpu",
    )
    input_tensor_fp32.stop_gradient = False
    input_tensor = paddle.cast(input_tensor_fp32, dtype=attr.src_dtype)
    t_tensor = paddle.cast(t_tensor_fp32, dtype=attr.tgt_dtype)
    out = paddle.cast(input_tensor, dtype=attr.tgt_dtype)
    out_grads = paddle.grad(
        [out], [input_tensor], [t_tensor]
    )
    assert(len(out_grads) == 1)
    return paddle.cast(out, "float32").numpy(), paddle.cast(out_grads[0], "float32").numpy()


