// Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/array_kernel_arg_value.h"

#include "phd/port.h"
#include "phd/test.h"

namespace gpu {
namespace cldrive {
namespace {

TEST(ArrayKernelArgValue, IntValuesAreEqual) {
  ArrayKernelArgValue<phd::int32> a(5, 0);
  ArrayKernelArgValue<phd::int32> b(5, 0);
  EXPECT_EQ(a, &b);
}

TEST(ArrayKernelArgValue, DifferentIntValuesAreNotEqual) {
  ArrayKernelArgValue<phd::int32> a(5, 0);
  ArrayKernelArgValue<phd::int32> b(5, -1);
  EXPECT_NE(a, &b);
}

TEST(ArrayKernelArgValue, DifferentSizeArraysAreNotEqual) {
  ArrayKernelArgValue<phd::int32> a(5, 0);
  ArrayKernelArgValue<phd::int32> b(4, 0);
  EXPECT_NE(a, &b);
}

TEST(ArrayKernelArgValue, VectorSizeFive) {
  ArrayKernelArgValue<phd::int32> a(5, 0);
  EXPECT_EQ(a.size(), 5);
}

TEST(ArrayKernelArgValue, VectorSizeTen) {
  ArrayKernelArgValue<phd::int32> a(10, 0);
  EXPECT_EQ(a.size(), 10);
}

TEST(ArrayKernelArgValue, FloatValuesAreEqual) {
  ArrayKernelArgValue<float> a(5, 0.5);
  ArrayKernelArgValue<float> b(5, 0.5);
  EXPECT_EQ(a, &b);
}

TEST(ArrayKernelArgValue, FloatValuesAreNotEqual) {
  ArrayKernelArgValue<float> a(5, 0.5);
  ArrayKernelArgValue<float> b(5, -0.5);
  EXPECT_NE(a, &b);
}

}  // anonymous namespace
}  // namespace cldrive
}  // namespace gpu

TEST_MAIN();
