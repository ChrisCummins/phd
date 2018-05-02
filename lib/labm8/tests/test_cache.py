# Copyright (C) 2015-2017 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
from unittest import main
from tests import TestCase

from labm8 import fs
from labm8 import system
from labm8 import cache

class TestCache(TestCase):
    def _test_cache(self, _cache):
        _cache.clear()

        # Item setter
        _cache["foo"] = 1
        _cache["bar"] = 2

        # in operator
        self._test(True, "foo" in _cache)
        self._test(True, "bar" in _cache)

        # Item getter
        self._test(1, _cache["foo"])
        self._test(2, _cache["bar"])

        # Lookup error
        self._test(False, "notakey" in _cache)
        self.assertRaises(KeyError, _cache.__getitem__, "notakey")

        # get() method
        self._test(1, _cache.get("foo"))
        self._test(2, _cache.get("bar"))

        # get() method default
        self._test(None, _cache.get("baz"))
        self._test(10, _cache.get("baz", 10))

        # "del" operator
        del _cache["bar"]
        self._test(False, "baz" in _cache)

        _cache.clear()

    # Cache
    def test_cache(self):
        # Test interface.
        _cache = cache.Cache()
        self.assertRaises(NotImplementedError, _cache.get, "foo")
        self.assertRaises(NotImplementedError, _cache.clear)
        self.assertRaises(NotImplementedError, _cache.items)
        self.assertRaises(NotImplementedError, _cache.__getitem__, "foo")
        self.assertRaises(NotImplementedError, _cache.__setitem__, "foo", 1)
        self.assertRaises(NotImplementedError, _cache.__contains__, "foo")
        self.assertRaises(NotImplementedError, _cache.__delitem__, "foo")

    # TransientCache
    def test_transient_cache(self):
        _cache = cache.TransientCache()
        self._test_cache(_cache)

        # Test copy constructor.
        _cache["foo"] = 1
        _cache["bar"] = 2
        _cache["baz"] = 3

        cache2 = cache.TransientCache(_cache)
        self._test(1, cache2["foo"])
        self._test(2, cache2["bar"])
        self._test(3, cache2["baz"])

        self._test(1, _cache["foo"])
        self._test(2, _cache["bar"])
        self._test(3, _cache["baz"])

    # JsonCache
    def test_json_cache(self):
        # Load test-set
        _cache = cache.JsonCache("tests/data/jsoncache.json")

        self._test(True, "foo" in _cache)
        self._test(1, _cache["foo"])
        self._test_cache(_cache)

        # Test copy constructor.
        _cache["foo"] = 1
        _cache["bar"] = 2
        _cache["baz"] = 3

        cache2 = cache.JsonCache("/tmp/labm8.cache.json", _cache)
        self._test(1, cache2["foo"])
        self._test(2, cache2["bar"])
        self._test(3, cache2["baz"])

        self._test(1, _cache["foo"])
        self._test(2, _cache["bar"])
        self._test(3, _cache["baz"])

        _cache.clear()

        # Set for next time.
        _cache["foo"] = 1
        _cache.write()


class TestFSCache(TestCase):
    def test_init_and_empty(self):
        c = cache.FSCache("/tmp/labm8-cache-init-and-empty")
        self.assertTrue(fs.isdir("/tmp/labm8-cache-init-and-empty"))
        c.clear()
        self.assertFalse(fs.isdir("/tmp/labm8-cache-init-and-empty"))

    def test_set_and_get(self):
        fs.rm("/tmp/labm8-cache-set-and-get")
        c = cache.FSCache("/tmp/labm8-cache-set-and-get")

        # create file
        system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
        # sanity check
        self.assertEqual(fs.read("/tmp/labm8.testfile.txt"),
                        ["Hello, world!"])

        # insert file into cache
        c['foobar'] = "/tmp/labm8.testfile.txt"

        # file must be in cache
        self.assertTrue(fs.isfile(c.keypath("foobar")))
        # file must have been moved
        self.assertFalse(fs.isfile("/tmp/labm8.testfile.txt"))
        # check file contents
        self.assertTrue(fs.read(c['foobar']), ["Hello, world!"])
        self.assertEqual(fs.read(c['foobar']), fs.read(c.get('foobar')))
        c.clear()

    def test_404(self):
        c = cache.FSCache("/tmp/labm8-cache-404")
        with self.assertRaises(KeyError):
            a = c['foobar']
        with self.assertRaises(KeyError):
            del c['foobar']
        self.assertEqual(c.get("foobar"), None)
        self.assertEqual(c.get("foobar", 5), 5)
        c.clear()

    def test_remove(self):
        c = cache.FSCache("/tmp/labm8-cache-remove")

        # create file
        system.echo("Hello, world!", "/tmp/labm8.test.remove.txt")
        # sanity check
        self.assertEqual(fs.read("/tmp/labm8.test.remove.txt"),
                         ["Hello, world!"])

        # insert file into cache
        c['foobar'] = "/tmp/labm8.test.remove.txt"

        # sanity check
        self.assertEqual(fs.read(c['foobar']), ["Hello, world!"])

        # remove from cache
        del c['foobar']

        with self.assertRaises(KeyError):
            a = c['foobar']
        self.assertEqual(c.get("foobar"), None)
        c.clear()

    def test_dict_key(self):
        c = cache.FSCache("/tmp/labm8-cache-dict")

        # create file
        system.echo("Hello, world!", "/tmp/labm8.test.remove.txt")
        # sanity check
        self.assertEqual(fs.read("/tmp/labm8.test.remove.txt"),
                         ["Hello, world!"])

        # insert file into cache
        key = {'a': 5, "c": [1, 2, 3]}
        c[key] = "/tmp/labm8.test.remove.txt"

        # check file contents
        self.assertTrue(fs.read(c[key]),
                        ["Hello, world!"])
        c.clear()

    def test_missing_key(self):
        c = cache.FSCache("/tmp/labm8-missing-key")

        with self.assertRaises(ValueError):
            c['foo'] = '/not/a/real/path'

        c.clear()

    def test_iter_len(self):
        c = cache.FSCache("/tmp/labm8-fscache-iter",
                          escape_key=cache.escape_path)
        c.clear()

        system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
        c["foo"] = "/tmp/labm8.testfile.txt"

        for path in c:
            self.assertEqual(path, c.keypath("foo"))

        system.echo("Hello, world!", "/tmp/labm8.testfile.txt")
        c["bar"] = "/tmp/labm8.testfile.txt"

        self.assertEqual(len(c), 2)

        self.assertEqual(len(c.ls()), 2)
        self.assertTrue("bar" in c.ls())
        self.assertTrue("foo" in c.ls())

        c.clear()
