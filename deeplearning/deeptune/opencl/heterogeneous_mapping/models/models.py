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
"""Models for predicting heterogeneous device mapping.

Attributes:
  ALL_MODELS: A set of HeterogeneousMappingModel subclasses.
"""
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import deeptune
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import grewe
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import lda
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import (
  static_mapping,
)
from labm8.py import app
from labm8.py import labtypes

FLAGS = app.FLAGS

# Import all models into the module namespace. This is for convenience of
# calling code, but is also required for ALL_MODELS to find the subclass.
DeepTune = deeptune.DeepTune
DeepTuneInst2Vec = ncc.DeepTuneInst2Vec
Grewe = grewe.Grewe
HeterogeneousMappingModel = base.HeterogeneousMappingModel
Lda = lda.Lda
StaticMapping = static_mapping.StaticMapping

# All models.
ALL_MODELS = labtypes.AllSubclassesOfClass(base.HeterogeneousMappingModel)
