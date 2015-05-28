# Copyright (C) 2015 Chris Cummins.
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

import labm8 as lab
from labm8 import cache

class TestCache(TestCase):

    def _test_cache(self, _cache):
        _cache.clear()

        # Item setter
        _cache["foo"] = 1
        _cache["bar"] = 2

        # "in" keyword
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

        _cache.clear()

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

        cache2 = cache.TransientCache(_cache)
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


if __name__ == '__main__':
    main()
