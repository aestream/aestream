import datetime
import time

from aestream import FileInput

# Reads events from the example file, specifying it's shape (346, 260)
# By default, we send the tensors to the CPU
#   - if you have a GPU, try changing this to "cuda"
with FileInput("davis.aedat4", (346, 260), device="cpu") as stream:

    # In this case, we read() every 100ms
    interval = 0.1
    t_0 = time.time()

    # Loop forever
    # stream.stop_stream()
    while stream.is_streaming():
        # When 500 ms passed...
        if t_0 + interval <= time.time():

            # Grab a tensor of the events arriving during the past 500ms
            frame = stream.read()

            # Reset the time so we're again counting to 500ms
            t_0 = time.time()

            # Sum the incoming events and print along the timestamp
            time_string = datetime.datetime.fromtimestamp(t_0).time()
            print(f"Frame at {time_string} with {frame.sum()} events")
