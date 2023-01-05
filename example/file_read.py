import datetime
import time

from aestream import FileInput

# Reads events from the example file, specifying it's shape (346, 260)
# By default, we send the tensors to the CPU with Numpy
#   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
s = 0
f = FileInput("sample.dat", (640, 480))
st = time.time_ns()
arr = f.events_co()
print((time.time_ns() - st) / 1e6, len(arr))
