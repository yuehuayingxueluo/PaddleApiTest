import numpy as np

def generate_np_inputs_and_dout():
    np.random.seed(0)
    logits_t = np.random.random(size=[1, 46256]).astype("float32")
    logits = logits_t - 0.5
    label = np.random.randint(46256, size=[1]).astype("int64")
    dout = np.random.random(size=[1]).astype("float32")

    np.savez("./inputs_case1.npz", logits=logits, label=label, dout=dout)