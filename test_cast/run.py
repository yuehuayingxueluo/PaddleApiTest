import os
import argparse
import itertools
import logging
from common import io
from common import framework
import attribute
from attribute import CastAttr
import numpy as np

class CastRuntime(framework.Runtime):
  def __init__(self, mode, attr):
    sup = super(CastRuntime, self)
    if mode == "paddle_eager":
      import paddle_runtime
      sup.__init__(mode, attr, paddle_runtime.Cast.launch_eager)
    elif mode == "paddle_static":
      import paddle_runtime
      sup.__init__(mode, attr, paddle_runtime.Cast.launch_static)
    elif mode == "torch_eager":
      import torch_runtime
      sup.__init__(mode, attr, torch_runtime.Cast.launch_eager)

def parse_args():
  parser = argparse.ArgumentParser(description="test")
  parser.add_argument("--tag", type=str) # paddle_dev, paddle_rel, torch
  parser.add_argument("--input_path", type=str, default="inputs.pkl")
  parser.add_argument("--output_dir", type=str, default="./")
  args = parser.parse_args()
  return args.tag, args.input_path, args.output_dir


if __name__ == "__main__":
  tag, input_path, output_path = parse_args()
  logging.warning("-- tag={}".format(tag))
  logging.warning("-- input_path={}".format(input_path))
  logging.warning("-- output_path={}".format(output_path))

  feeds = io.Pickle.load(input_path)
  logging.warning("-- feed data length={}".format(len(feeds)))

  result = list()

  attrs = (
    CastAttr("float32", "float16"),
    CastAttr("float16", "float32"),
    CastAttr("uint16", "float32"),
    CastAttr("float32", "uint16"),
    CastAttr("uint16", "float16"),
    CastAttr("float16", "uint16"),
  )

  if (tag.startswith("paddle")):
    modes = ("paddle_eager", "paddle_static")
  else:
    modes = ("torch_eager", "torch_eager")

  for (feed, attr, mode) in itertools.product(feeds, attrs, modes):
    runtime = CastRuntime(mode, attr)
    runtime.stability_test(feed)
    result.append(runtime.dispatch(feed))

  io.Pickle.save(result, os.path.join(output_path, tag + ".pkl"))

