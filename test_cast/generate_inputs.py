import attribute
from common import io

def generate_inputs(shape, len):
  ret = list()
  for i in range(len):
    ret.append(attribute.CastRandomInput(shape))
  return ret

if __name__ == "__main__":
  inputs = generate_inputs([1, 1], 20)
  path = "inputs.pkl"
  io.Pickle.save(inputs, path)
