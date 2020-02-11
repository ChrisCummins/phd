#include "deeplearning/ml4pl/graphs/graphviz_converter.h"
#include "deeplearning/ml4pl/graphs/programl.pb.h"
#include "deeplearning/ml4pl/graphs/random_graph_builder.h"
#include "labm8/cpp/app.h"
#include "labm8/cpp/logging.h"

static const char* usage = "Generate a random program graph proto.";

DEFINE_int32(node_count, 0,
             "The number of nodes to create in the random graph.");

DEFINE_string(stdout_fmt, "pbtxt",
              "The type of output format to use. Valid options are: "
              "\"pb\" which prints binary protocol buffer, \"pbtxt\" which "
              "prints a text format protocol buffer, or \"dot\" which prints a "
              "graphviz dot string.");

// Assert that the stdout format is legal.
static bool ValidateStdoutFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt" || value == "dot") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt,dot";
  return false;
}
DEFINE_validator(stdout_fmt, &ValidateStdoutFormat);

int main(int argc, char** argv) {
  labm8::InitApp(&argc, &argv, usage);

  ml4pl::RandomGraphBuilder builder;

  const auto graphOr = builder.FastCreateRandom(FLAGS_node_count);
  if (!graphOr.ok()) {
    LOG(FATAL) << graphOr.status().ToString();
  }

  if (FLAGS_stdout_fmt == "pb") {
    graphOr.ValueOrDie().SerializeToOstream(&std::cout);
  } else if (FLAGS_stdout_fmt == "pbtxt") {
    std::cout << graphOr.ValueOrDie().DebugString();
  } else if (FLAGS_stdout_fmt == "dot") {
    ml4pl::SerializeGraphVizToString(graphOr.ValueOrDie(), &std::cout);
  } else {
    LOG(FATAL) << "unreachable";
  }

  return 0;
}
