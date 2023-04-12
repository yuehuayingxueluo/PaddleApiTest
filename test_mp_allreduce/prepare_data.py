import numpy as np

dim_1 = 4096
dim_2 = 12288

def generate_np_inputs_and_dout():
    x_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")
    dout_case1 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")

    x_case2 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")
    dout_case2 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")

    x_case3 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")
    dout_case3 = np.random.random(size=[1, dim_1, dim_2]).astype("float32")

    x_case4 = np.random.random(size=[1, dim_2]).astype("float32")
    dout_case4 = np.random.random(size=[1, dim_2]).astype("float32")

    x_case5 = np.random.random(size=[1, dim_2]).astype("float32")
    dout_case5 = np.random.random(size=[1, dim_2]).astype("float32")

    x_case6 = np.random.random(size=[1, dim_2]).astype("float32")
    dout_case6 = np.random.random(size=[1, dim_2]).astype("float32")

    np.savez("./1_inputs_case1.npz", x = x_case1, dout = dout_case1)
    np.savez("./1_inputs_case2.npz", x = x_case2, dout = dout_case2)
    np.savez("./1_inputs_case3.npz", x = x_case3, dout = dout_case3)

    np.savez("./2_inputs_case1.npz", x = x_case4, dout = dout_case4)
    np.savez("./2_inputs_case2.npz", x = x_case5, dout = dout_case5)
    np.savez("./2_inputs_case3.npz", x = x_case6, dout = dout_case6)


generate_np_inputs_and_dout()