"""Models for learning reachability analysis."""

import typing

from absl import flags

from experimental.compilers.reachability import control_flow_graph as cfg


FLAGS = flags.FLAGS


class ReachabilityModelBase(object):

  def Fit(self, training_graphs: typing.Iterator[cfg.ControlFlowGraph],
          validation_graphs: typing.Iterator[cfg.ControlFlowGraph]):
    raise NotImplementedError

  def Predict(self, testing_graphs: typing.Iterator[cfg.ControlFlowGraph]):
    raise NotImplementedError


class SequentialReachabilityModel(ReachabilityModelBase):
  pass
