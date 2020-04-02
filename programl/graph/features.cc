#include "programl/graph/features.h"

namespace programl {
namespace graph {

Feature CreateFeature(int64_t value) {
  Feature feature;
  feature.mutable_int64_list()->add_value(value);
  return feature;
}

Feature CreateFeature(const vector<int64_t> &value) {
  Feature feature;
  for (auto v : value) {
    feature.mutable_int64_list()->add_value(v);
  }
  return feature;
}

void SetFeature(Features *features, const char *label, const Feature &value) {
  (*features->mutable_feature())[label] = value;
}

}  // namespace graph
}  // namespace programl