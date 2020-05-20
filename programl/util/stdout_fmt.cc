// Copyright 2019-2020 the ProGraML authors.
//
// Contact Chris Cummins <chrisc.101@gmail.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "programl/util/stdout_fmt.h"

DEFINE_string(stdout_fmt, "pbtxt",
              "The type of output format to use. Valid options are: "
              "\"pbtxt\" which prints a text format protocol buffer, "
              "or \"pb\" which a binary format protocol buffer.");

// Assert that the stdout format is legal.
static bool ValidateStdoutFormat(const char* flagname, const string& value) {
  if (value == "pb" || value == "pbtxt") {
    return true;
  }

  LOG(FATAL) << "Unknown --" << flagname << ": `" << value << "`. Supported "
             << "formats: pb,pbtxt";
  return false;
}
DEFINE_validator(stdout_fmt, &ValidateStdoutFormat);