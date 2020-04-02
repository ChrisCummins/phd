from programl.graph.analysis.py import analysis_pybind
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


def RunAnalysis(
  analysis: str, graph: program_graph_pb2.ProgramGraph
) -> program_graph_features_pb2.ProgramGraphFeaturesList:
  graph_features = program_graph_features_pb2.ProgramGraphFeaturesList()
  serialized_graph_features = analysis_pybind.RunAnalysis(
    analysis, graph.SerializeToString()
  )
  graph_features.ParseFromString(serialized_graph_features)
  return graph_features
