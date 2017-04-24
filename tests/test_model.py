#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
from unittest import TestCase, skip, skipIf, main
import tests

from io import StringIO
from labm8 import crypto
from labm8 import fs
from six import string_types
from tempfile import mkdtemp

import clgen
from clgen import cache


def get_test_model(vocab="char"):
    return clgen.Model.from_json({
        "corpus": {
            "path": tests.data_path("tiny", "corpus"),
            "vocabulary": vocab
        },
        "architecture": {
          "rnn_size": 8,
          "num_layers": 2
        },
        "train_opts": {
            "epochs": 1
        }
    })


class TestModel(TestCase):
    def test_hash(self):
        m1 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            }
        })

        # same as m1, with explicit default opt:
        m2 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": True
            }
        })

        # different opt value:
        m3 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": False
            }
        })

        self.assertEqual(m1.hash, m2.hash)
        self.assertNotEqual(m2.hash, m3.hash)

    def test_checkpoint_path_untrained(self):
        m = get_test_model()
        m.cache.clear()  # untrain
        self.assertEqual(m.checkpoint_path, None)

    def test_eq(self):
        m1 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": False
            }
        })
        m2 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": False
            }
        })
        m3 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": True
            }
        })

        self.assertEqual(m1, m2)
        self.assertNotEqual(m2, m3)
        self.assertNotEqual(m1, False)
        self.assertNotEqual(m1, 'abcdef')

    def test_to_json(self):
        m1 = clgen.Model.from_json({
            "corpus": {
                "path": tests.data_path("tiny", "corpus")
            },
            "train_opts": {
                "intermediate_checkpoints": True
            }
        })
        m2 = clgen.Model.from_json(m1.to_json())
        self.assertEqual(m1, m2)


if __name__ == "__main__":
    main()
