"""Unit tests for //labm8:cache."""
import tempfile

import pytest
from absl import flags

from labm8 import cache
from labm8 import fs
from labm8 import system
from labm8 import test

FLAGS = flags.FLAGS


def _TestCacheOps(_cache):
  _cache.clear()

  # Item setter
  _cache["foo"] = 1
  _cache["bar"] = 2

  # in operator
  assert "foo" in _cache
  assert "bar" in _cache

  # Item getter
  assert 1 == _cache["foo"]
  assert 2 == _cache["bar"]

  # Lookup error
  assert "notakey" not in _cache
  assert pytest.raises(KeyError, _cache.__getitem__, "notakey")

  # get() method
  assert 1 == _cache.get("foo")
  assert 2 == _cache.get("bar")

  # get() method default
  assert not _cache.get("baz")
  assert 10 == _cache.get("baz", 10)

  # "del" operator
  del _cache["bar"]
  assert "baz" not in _cache

  _cache.clear()


def test_cache():
  # Test abstract interface.
  c = cache.Cache()
  assert pytest.raises(NotImplementedError, c.get, "foo")
  assert pytest.raises(NotImplementedError, c.clear)
  assert pytest.raises(NotImplementedError, c.items)
  assert pytest.raises(NotImplementedError, c.__getitem__, "foo")
  assert pytest.raises(NotImplementedError, c.__setitem__, "foo", 1)
  assert pytest.raises(NotImplementedError, c.__contains__, "foo")
  assert pytest.raises(NotImplementedError, c.__delitem__, "foo")


def test_TransientCache():
  _cache = cache.TransientCache()
  _TestCacheOps(_cache)

  # Test copy constructor.
  _cache["foo"] = 1
  _cache["bar"] = 2
  _cache["baz"] = 3

  cache2 = cache.TransientCache(_cache)
  assert 1 == cache2["foo"]
  assert 2 == cache2["bar"]
  assert 3 == cache2["baz"]

  assert 1 == _cache["foo"]
  assert 2 == _cache["bar"]
  assert 3 == _cache["baz"]


def test_JsonCache():
  with tempfile.NamedTemporaryFile(prefix='labm8_') as f:
    # Load test-set
    fs.cp('labm8/data/test/jsoncache.json', f.name)
    _cache = cache.JsonCache(f.name)

    assert "foo" in _cache
    assert 1 == _cache["foo"]
    _TestCacheOps(_cache)

    # Test copy constructor.
    _cache["foo"] = 1
    _cache["bar"] = 2
    _cache["baz"] = 3

    with tempfile.NamedTemporaryFile(prefix='labm8_') as f2:
      cache2 = cache.JsonCache(f2.name, _cache)
      assert 1 == cache2["foo"]
      assert 2 == cache2["bar"]
      assert 3 == cache2["baz"]
      assert 1 == _cache["foo"]
      assert 2 == _cache["bar"]
      assert 3 == _cache["baz"]
      _cache.clear()
      # Set for next time.
      _cache["foo"] = 1
      _cache.write()


def test_FSCache_init_and_empty():
  c = cache.FSCache("/tmp/labm8-cache-init-and-empty")
  assert fs.isdir("/tmp/labm8-cache-init-and-empty")
  c.clear()
  assert not fs.isdir("/tmp/labm8-cache-init-and-empty")


def test_set_and_get():
  fs.rm("/tmp/labm8-cache-set-and-get")
  c = cache.FSCache("/tmp/labm8-cache-set-and-get")
  # create file
  system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
  # sanity check
  assert fs.read("/tmp/labm8.testfile.txt") == ["Hello, world!"]
  # insert file into cache
  c['foobar'] = "/tmp/labm8.testfile.txt"
  # file must be in cache
  assert fs.isfile(c.keypath("foobar"))
  # file must have been moved
  assert not fs.isfile("/tmp/labm8.testfile.txt")
  # check file contents
  assert fs.read(c['foobar']) == ["Hello, world!"]
  assert fs.read(c['foobar']) == fs.read(c.get('foobar'))
  c.clear()


def test_FSCache_404():
  c = cache.FSCache("/tmp/labm8-cache-404")
  with pytest.raises(KeyError):
    c['foobar']
  with pytest.raises(KeyError):
    del c['foobar']
  assert not c.get("foobar")
  assert c.get("foobar", 5) == 5
  c.clear()


def test_FSCache_remove():
  c = cache.FSCache("/tmp/labm8-cache-remove")
  # create file
  system.echo("Hello, world!", "/tmp/labm8.test.remove.txt")
  # sanity check
  assert fs.read("/tmp/labm8.test.remove.txt") == ["Hello, world!"]
  # insert file into cache
  c['foobar'] = "/tmp/labm8.test.remove.txt"
  # sanity check
  assert fs.read(c['foobar']) == ["Hello, world!"]
  # remove from cache
  del c['foobar']
  with pytest.raises(KeyError):
    c['foobar']
  assert not c.get("foobar")
  c.clear()


def test_FSCache_dict_key():
  c = cache.FSCache("/tmp/labm8-cache-dict")
  # create file
  system.echo("Hello, world!", "/tmp/labm8.test.remove.txt")
  # sanity check
  assert fs.read("/tmp/labm8.test.remove.txt") == ["Hello, world!"]
  # insert file into cache
  key = {'a': 5, "c": [1, 2, 3]}
  c[key] = "/tmp/labm8.test.remove.txt"
  # check file contents
  assert fs.read(c[key]) == ["Hello, world!"]
  c.clear()


def test_FSCache_missing_key():
  c = cache.FSCache("/tmp/labm8-missing-key")
  with pytest.raises(ValueError):
    c['foo'] = '/not/a/real/path'
  c.clear()


def test_FSCache_iter_len():
  c = cache.FSCache("/tmp/labm8-fscache-iter", escape_key=cache.escape_path)
  c.clear()
  system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
  c["foo"] = "/tmp/labm8.testfile.txt"
  for path in c:
    assert path == c.keypath("foo")
  system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
  c["bar"] = "/tmp/labm8.testfile.txt"
  assert len(c) == 2
  assert len(c.ls()) == 2
  assert "bar" in c.ls()
  assert "foo" in c.ls()
  c.clear()


if __name__ == '__main__':
  test.Main()
