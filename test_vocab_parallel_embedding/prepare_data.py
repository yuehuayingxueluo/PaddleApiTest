import numpy as np

def generate_np_inputs_and_dout():
    np.random.seed(0)
    dim_1 = 56200
    dim_2 = 4096
    dim_3 = 12288
    x_case1 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case1 = np.random.random(size=[dim_1, dim_3]).astype("float32")
    dout_case1 = np.random.random(size=[1, dim_2, dim_3]).astype("float32")

    x_case2 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case2 = np.random.random(size=[dim_1, dim_3]).astype("float32")
    dout_case2 = np.random.random(size=[1, dim_2, dim_3]).astype("float32")

    x_case3 = np.random.randint(low=0, high=dim_1, size=[1, dim_2]).astype("int64")
    table_case3 = np.random.random(size=[dim_1, dim_3]).astype("float32")
    dout_case3 = np.random.random(size=[1, dim_2, dim_3]).astype("float32")

    np.savez("./inputs_case1.npz", x = x_case1, table = table_case1, dout = dout_case1)
    np.savez("./inputs_case2.npz", x = x_case2, table = table_case1, dout = dout_case2)
    np.savez("./inputs_case3.npz", x = x_case3, table = table_case1, dout = dout_case3)