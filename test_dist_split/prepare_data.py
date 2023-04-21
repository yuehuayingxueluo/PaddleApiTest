import numpy as np

def generate_np_inputs_and_dout():
    np.random.seed(0)

    dim_1 = 4096
    dim_2 = 12288
    dim_3 = 24576
    dim_4 = 49152
    dim_5 = 6144


    x_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5
    weight_case1 = np.random.random(size=[dim_2, dim_2]).astype("float32") - 0.5
    axis_case1 = np.array([1])
    bias_case1 = np.random.random(size=[dim_2]).astype("float32") - 0.5
    dout_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5

    x_case2 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5
    weight_case2 = np.random.random(size=[dim_2, dim_3]).astype("float32") - 0.5
    axis_case1 = np.array([1])
    bias_case2 = np.random.random(size=[dim_3]).astype("float32") - 0.5
    dout_case2 = np.random.random(size=[1, dim_1, dim_3]).astype("float32") - 0.5

    x_case3 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5
    weight_case3 = np.random.random(size=[dim_2, dim_4]).astype("float32") - 0.5
    axis_case1 = np.array([1])
    bias_case3 = np.random.random(size=[dim_4]).astype("float32") - 0.5
    dout_case3 = np.random.random(size=[1, dim_1, dim_4]).astype("float32") - 0.5

    x_case4 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5
    weight_case4 = np.random.random(size=[dim_2, dim_2]).astype("float32") - 0.5
    axis_case1 = np.array([0])
    bias_case4 = np.random.random(size=[dim_2]).astype("float32") - 0.5
    dout_case4 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5

    x_case5 = np.random.random(size=[1, dim_1, dim_4]).astype("float32") - 0.5
    weight_case5 = np.random.random(size=[dim_4, dim_2]).astype("float32") - 0.5
    axis_case1 = np.array([0])
    bias_case5 = np.random.random(size=[dim_2]).astype("float32") - 0.5
    dout_case5 = np.random.random(size=[1, dim_1, dim_2]).astype("float32") - 0.5

    np.savez("./inputs_case1.npz", x=x_case1, weight=weight_case1, axis=axis_case1, bias=bias_case1, dout=dout_case1)
    np.savez("./inputs_case2.npz", x=x_case2, weight=weight_case2, axis=axis_case1, bias=bias_case2, dout=dout_case2)
    np.savez("./inputs_case3.npz", x=x_case3, weight=weight_case3, axis=axis_case1, bias=bias_case3, dout=dout_case3)
    np.savez("./inputs_case4.npz", x=x_case4, weight=weight_case4, axis=axis_case1, bias=bias_case4, dout=dout_case4)
    np.savez("./inputs_case5.npz", x=x_case5, weight=weight_case5, axis=axis_case1, bias=bias_case5, dout=dout_case5)