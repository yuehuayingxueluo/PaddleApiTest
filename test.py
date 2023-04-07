import paddle
import torch
import numpy as np
from utils import convert_float_to_uint16,convert_uint16_to_float
from paddle.fluid.layers.utils import map_structure
# paddle.enable_static()
for i in range(1):
    np_x = np.random.random(size=[100]).astype(np.float32)
    np_y = np.random.random(size=[100]).astype(np.float32)
    # np_y = np.ones(shape=[1,1]).astype(np.float32)
    # print(np_x)
    # print(np_y)
    # np_x = convert_float_to_uint16(np_x)
    # print(np_x.dtype == np.dtype('uint16'))
    # np_y = convert_float_to_uint16(np_y)
    # print(np_x)
    # print(np_y)
    # y = paddle.to_tensor(np_y,dtype="bfloat16",place="gpu")
    # x = paddle.to_tensor(np_x,dtype="bfloat16",place="gpu")
    # print(paddle.matmul(x,y))
    x = torch.tensor(np_x,device='cuda',dtype=torch.float32,requires_grad = True)
    y = torch.tensor(np_y,device='cuda',dtype=torch.float32,requires_grad = True)
    x_t = x.to(dtype=torch.bfloat16)
    y_t = y.to(dtype=torch.bfloat16)
    out = torch.matmul(x, y)
    out_grads = torch.autograd.grad([out], [x, y])
    
    # print(out_t)
    # print(out_t.float())
    # mp,sp=paddle.static.Program(),paddle.static.Program()
    # with paddle.static.program_guard(mp,sp):
    #     x = paddle.static.data('x',shape=[1],dtype="uint16")
    #     y = paddle.static.data('y',shape=[1],dtype="uint16")
    #     z = paddle.matmul(x,y)
    # exe = paddle.static.Executor(place=paddle.CUDAPlace(0))
    # out = exe.run(mp, feed={"x": np_x, "y" : np_y}, fetch_list=[z])[0]
    # print(out)
    # print(convert_uint16_to_float(out))

    # np_x = convert_float_to_uint16(np_x)
    # np_y = convert_float_to_uint16(np_y)
    # y = paddle.to_tensor(np_y,dtype="bfloat16",place="gpu")
    # x = paddle.to_tensor(np_x,dtype="bfloat16",place="gpu")
    # out_pd = paddle.matmul(x,y).numpy()
    # print("out_pd",out_pd)
    # out_pd = convert_uint16_to_float(out_pd)
    # # print(out_pd)
    # # print(convert_uint16_to_float(out_pd))
    # np.testing.assert_allclose(out_t,out_pd,atol=1e-3,rtol=1e-3)

# dtype = "float32"
# print('compare matmul eager forward res with torch failed in %s'%dtype)