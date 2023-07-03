import re

# Base class code
base_class_code = """
class TestSquareDevelopCase{}_FP32(TestSquareDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float32"
        self.size = [{}]

class TestSquareDevelopCase{}_FP16(TestSquareDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [{}]

class TestSquareDevelopCase{}_BFP16(TestSquareDevelopCase8_FP32):
    def init_params(self):
        self.dtype = "float16"
        self.size = [{}]
"""

# Sizes
sizes = [

[12528, 14336],
[14336, 5376],
[14336, 9632],
[14336],
[1792, 14336],
[4816, 14336], 
[5376], 
[9632],

]


# Generate code for each size
index = 9
for size in sizes:
    size_str = ", ".join(str(s) for s in size)
    generated_code = re.sub(r"\{\}", size_str, base_class_code.format(index, size_str, index, size_str, index, size_str, index, size_str))
    print(generated_code)
    print(" ")
    index += 1

