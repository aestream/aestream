import datetime
import time

from aestream import FileInput

# Reads events from the example file, specifying it's shape (346, 260)
# By default, we send the tensors to the CPU with Numpy
#   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
s = 0
f = FileInput("sample.dat", (640, 480))
# for e in f:
#     print(e)
st = time.time_ns()
parts = f.parts(1000)
print(parts)
p = next(parts)
print(len(p), type(p), p.dtype)
print(p[88])

s = len(p)
for e in parts:
    s += len(e)

print(s)
# print((time.time_ns() - st) / 1e6, arr)
