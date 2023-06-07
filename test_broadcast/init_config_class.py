import numpy as np
import sys
sys.path.append("..")
from utils import TOLERANCE, convert_dtype_to_torch_type

class InitConfigClass:
    def __init__(self):
        self._init_params()
        self._init_threshold()
        self._init_np_inputs_and_dout()

    def _init_params(self, np_input_dir="", dtype=""):
        self._np_input_dir = np_input_dir
        self._dtype = dtype
    
    def _init_threshold(self):
        self._atol = TOLERANCE[self._dtype]["atol"]
        self._rtol = TOLERANCE[self._dtype]["rtol"]

    def _init_np_inputs_and_dout(self):
        np_inputs_array = np.load(self._np_input_dir)
        # get np array from npz file
        self._np_x = np_inputs_array["x"]
        # convert np array dtype
        if self._dtype == "float16":
            self._np_x = self._np_x.astype("float16")