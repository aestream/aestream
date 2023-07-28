from aestream import FileInput

# Open a file while specifying it's shape (346, 260) and its device ("cpu")
# By default, we send the tensors to the CPU with Numpy
#   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
f = FileInput("sample.dat", (640, 480), device="cpu")

# Load the *entire* file into memory (be careful if your file is big)
events = f.load()

# Iterate over the events
for e in events:
    print(e)
