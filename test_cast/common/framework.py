from . import io

class Framework(object):
  @staticmethod
  def launch_eager(input_data, attr):
    pass

  @staticmethod
  def launch_static(input_data, attr):
    pass

class Attribute(object):
  def __init__():
    pass

class Runtime(object):
  def __init__(self, mode, attr, func):
    self.mode = mode
    self.attr = attr
    self.func = func
  
  def mode(self):
    return self.mode

  def attr(self):
    return self.attr

  def dispatch(self, input):
    return io.Result(self.mode, self.func(input, self.attr))

  def stability_test(self, input, rounds=10):
    res = self.dispatch(input)
    for i in range(rounds):
      self.dispatch(input).assert_equal(res)
