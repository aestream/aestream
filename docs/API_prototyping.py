# AEStream API Prototyping

## Ideas

from enum import Enum, auto


class BinningMode(Enum):
    n_timesteps = auto()
    n_events = auto()


mode = BinningMode.n_timesteps

# vs
'''
- `mode (str)` : Method for deciding how to bin events
    - `n_timesteps {default}` : Bins events in groups of `n` timesteps
    - `n_events` : Bins events in groups of `n` events
'''
mode = "n_timesteps"

#
# - Maybe group by source/sink instead
# - Maybe group by dense/sparse instead of frame/event
#

## Reader


class EventReader:

    def __init__(self, file):
        self.file = file
        self.events = read_file(file)


class FrameReader:

    def __init__(self, file):
        self.file = file
        self.frames = read_file(file)


## Iterator


class EventIterator:

    def __init__(
        self,
        file,
        start_timestep=0,
        end_timestep=None,
        mode="n_timesteps",
        n=1,
    ):
        self.file = file

    def __iter__(self):
        return self

    def __next__(self):
        return next_event()


class FrameIterator:

    def __init__(
        self,
        file,
        start_timestep=0,
        end_timestep=None,
        mode="n_timesteps",
        n=1,
    ):
        self.file = file

    def __iter__(self):
        return self

    def __next__(self):
        return next_frame()
