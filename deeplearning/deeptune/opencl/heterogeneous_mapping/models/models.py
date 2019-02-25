"""Models for predicting heterogeneous device mapping.

Attributes:
  ALL_MODELS: A set of HeterogeneousMappingModel subclasses.
"""

from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping.models import base
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import deeptune
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import grewe
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import lda
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import \
  static_mapping
from labm8 import labtypes

FLAGS = flags.FLAGS

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
