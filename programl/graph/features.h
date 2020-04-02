#pragma once

#include "programl/proto/features.pb.h"

#include <vector>

using std::vector;

namespace programl {
namespace graph {

Feature CreateFeature(int64_t value);

Feature CreateFeature(const vector<int64_t> &value);

void SetFeature(Features *features, const char *label, const Feature &value);

}  // namespace graph
}  // namespace programl