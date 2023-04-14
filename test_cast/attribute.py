import numpy as np
from common import framework

class CastAttr(framework.Attribute):
  def __init__(self, src_dtype, tgt_dtype):
    self.src_dtype = src_dtype
    self.tgt_dtype = tgt_dtype

class CastRandomInput(object):
  def __init__(self, shape, dtype="float32"):
    self.x = np.random.random(size=shape).astype(dtype)
    self.out_t = np.random.random(size=shape).astype(dtype)
