# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Common tests for heterogeneous mapping models."""
import pathlib
import tempfile

from labm8 import app

FLAGS = app.FLAGS


class HeterogeneousMappingModelTest(object):
  """Common tests for heterogeneous mapping models."""

  def __init__(self, model_class, df, atomizer, model_init_opts=None):
    self.model_class = model_class
    self.model_init_opts = model_init_opts or {}
    self.df = df
    self.atomizer = atomizer

    self.test_init()
    self.test_save_restore()
    self.test_train_predict()

  def test_init(self):
    """Test that init() can be called without error."""
    model = self.model_class(**self.model_init_opts)
    model.init(0, self.atomizer)

  def test_save_restore(self):
    """Test that models can be saved and restored from file."""
    with tempfile.TemporaryDirectory(prefix='phd_') as d:
      tempdir = pathlib.Path(d)

      model_to_file = self.model_class(**self.model_init_opts)
      model_to_file.init(0, self.atomizer)
      model_to_file.save(tempdir / 'model')

      model_from_file = self.model_class(**self.model_init_opts)
      model_from_file.restore(tempdir / 'model')
      # We can't test that restoring the model from file actually does anything,
      # since we don't have __eq__ operator implemented for models.

  def test_train_predict(self):
    """Test that models can be trained, and used to make predictions."""
    model = self.model_class(**self.model_init_opts)
    model.init(0, self.atomizer)
    model.train(self.df, 'amd_tahiti_7970')
    model.predict(self.df, 'amd_tahiti_7970')
