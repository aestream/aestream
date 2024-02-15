import pytest

from aestream import Backend, FileInput


def test_read_numpy():
    FileInput("example/sample.aedat4", shape=(600, 400)).read(Backend.Numpy)


def test_read_numpy_string():
    FileInput("example/sample.aedat4", shape=(600, 400)).read("Numpy")
    FileInput("example/sample.aedat4", shape=(600, 400)).read("numpy")


def test_read_nonexistent_backend():
    with pytest.raises(AttributeError):
        FileInput("example/sample.aedat4", shape=(600, 400)).read("NonexistentBackend")
