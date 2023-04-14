import numpy as np

def generate_np_inputs_and_dout():
    np.random.seed(0)
    dim_1 = 56200
    dim_2 = 4096
    dim_3 = 12288
    x_case1 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case1 = np.random.random(size=[dim_1, dim_3]).astype("float32")
    dout_case1 = np.random.random(size=[1, dim_2, dim_3]).astype("float32")

    np.savez("./inputs_case1.npz", x = x_case1, table = table_case1, dout = dout_case1)