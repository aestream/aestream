import datetime
import time

from aestream import FileInput

# Reads events from the example file, specifying it's shape (640, 480)
# By default, we send the tensors to the CPU with Numpy
#   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
s = 0
st = time.time()
with FileInput("sample.dat", (640, 480), device="cuda", ignore_time=True) as stream:
    # In this case, we read() every 1ms
    interval = 0.001
    c = 0
    t_0 = time.time()

    # Loop forever
    while stream.is_streaming():
        # When 1 ms passed...
        if t_0 + interval <= time.time():
            # Grab a tensor of the events arriving during the past 1ms
            frame = stream.read(backend="torch")

            # Sum up the events and increment frame counter
            s += frame.sum()
            c += 1

            # Reset the time so we're again counting to 1ms
            t_0 = time.time()

print(
    f"{c} frames of 1ms each read in {time.time() - st:.3}s containing a total of {s} events"
)
