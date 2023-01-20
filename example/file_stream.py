import datetime
import time

from aestream import FileInput

# Reads events from the example file, specifying it's shape (346, 260)
# By default, we send the tensors to the CPU with Numpy
#   - if you have a PyTorch installation with a GPU, try changing this to "cuda"
s = 0
st = time.time()
with FileInput("sample.dat", (640, 480), device="cpu", ignore_time=True) as stream:

    # In this case, we read() every 100ms
    interval = 0.01
    c = 0
    t_0 = time.time()

    # Loop forever
    while stream.is_streaming():
        # When 100 ms passed...
        if t_0 + interval <= time.time():

            # Grab a tensor of the events arriving during the past 100ms
            frame = stream.read()

            # Reset the time so we're again counting to 100ms
            t_0 = time.time()
            # s += frame.sum()
            # Sum the incoming events and print along the timestamp
            #time_string = datetime.datetime.fromtimestamp(t_0).time()
            #print(f"Frame at {time_string} with {frame.sum()} events")
            c +=1
    
    # s += stream.read().sum()
    print(s, time.time() - st)
    print(c)