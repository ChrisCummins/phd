"""Unit tests for //deeplearning/clgen:sample_observers."""
import pathlib

from deeplearning.clgen import sample_observers
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


def test_MaxSampleCountObserver():
  observer = sample_observers.MaxSampleCountObserver(3)
  assert observer.OnSample(None)
  assert observer.OnSample(None)
  assert not observer.OnSample(None)


def test_SaveSampleTextObserver(tempdir: pathlib.Path):
  observer = sample_observers.SaveSampleTextObserver(tempdir)
  contents = "Hello, world!"
  sample = model_pb2.Sample(text=contents)

  assert observer.OnSample(sample)
  path = tempdir / f"{crypto.sha256_str(contents)}.txt"
  assert path.is_file()
  assert fs.Read(path) == contents


def test_PrintSampleObserver(capsys):
  observer = sample_observers.PrintSampleObserver()
  sample = model_pb2.Sample(text="Hello, world!")

  assert observer.OnSample(sample)
  captured = capsys.readouterr()
  assert (
    captured.out
    == """\
=== CLGEN SAMPLE ===

Hello, world!

"""
  )


def test_InMemorySampleSaver():
  observer = sample_observers.InMemorySampleSaver()
  sample = model_pb2.Sample(text="Hello, world!")

  assert observer.OnSample(sample)
  assert len(observer.samples) == 1
  assert observer.samples[-1].text == "Hello, world!"

  assert observer.OnSample(sample)
  assert len(observer.samples) == 2
  assert observer.samples[-1].text == "Hello, world!"


if __name__ == "__main__":
  test.Main()
