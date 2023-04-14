import argparse
from common import io

def parse_args():
  parser = argparse.ArgumentParser(description="test")
  parser.add_argument("--paddle_rel", type=str, default="paddle_rel.pkl")
  parser.add_argument("--paddle_dev", type=str, default="paddle_dev.pkl")
  parser.add_argument("--torch", type=str, default="torch.pkl")
  args = parser.parse_args()
  return args.paddle_rel, args.paddle_dev, args.torch

if __name__ == "__main__":
  pd_rel_p, pd_dev_p, torch_p = parse_args()

  pd_rel_list = io.Pickle.load(pd_rel_p)
  pd_dev_list = io.Pickle.load(pd_dev_p)
  torch_list = io.Pickle.load(torch_p)

  assert(len(pd_dev_list) == len(pd_rel_list))
  assert(len(pd_dev_list) == len(torch_list))

  for pd_dev, pd_rel, torch in zip(pd_dev_list, pd_rel_list, torch_list):
    pd_dev.assert_equal(pd_rel)
    pd_dev.assert_equal(torch)
