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
#include "gpu/cldrive/kernel_arg.h"

#include "gpu/cldrive/global_memory_arg_value.h"
#include "gpu/cldrive/kernel_arg_value.h"
#include "gpu/cldrive/proto/cldrive.pb.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"
#include "gpu/cldrive/testutil.h"

#include "third_party/opencl/cl.hpp"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/test.h"

namespace gpu {
namespace cldrive {
namespace {

TEST(OpenClTypeFromString, Int) {
  auto arg_type = OpenClTypeFromString("int");
  ASSERT_TRUE(arg_type.ok());
  EXPECT_EQ(arg_type.ValueOrDie(), OpenClType::INT);
}

TEST(OpenClTypeFromString, Unknown) {
  EXPECT_FALSE(OpenClTypeFromString("unknown").ok());
}

class KernelArgTest : public ::testing::Test {
 protected:
  virtual void SetUp() override { context_ = cl::Context::getDefault(); }
  cl::Context context_;
};

TEST_F(KernelArgTest, UnsupportedType) {
  cl::Kernel kernel =
      test::CreateClKernel("kernel void A(read_only image2d_t a) {}");

  KernelArg arg;
  EXPECT_FALSE(arg.Init(&kernel, 0).ok());
}

TEST_F(KernelArgTest, GlobalPointerIsGlobal) {
  cl::Kernel kernel = test::CreateClKernel("kernel void A(global int* a) {}");

  KernelArg arg;
  ASSERT_TRUE(arg.Init(&kernel, 0).ok());
  EXPECT_TRUE(arg.IsGlobal());
}

TEST_F(KernelArgTest, GlobalPointerIsNotLocal) {
  cl::Kernel kernel = test::CreateClKernel("kernel void A(global int* a) {}");

  KernelArg arg;
  ASSERT_TRUE(arg.Init(&kernel, 0).ok());
  EXPECT_FALSE(arg.IsLocal());
}

TEST_F(KernelArgTest, TryToCreateOnesValueGlobalInt) {
  cl::Kernel kernel = test::CreateClKernel("kernel void A(global int* a) {}");

  KernelArg arg;
  ASSERT_TRUE(arg.Init(&kernel, 0).ok());

  auto arg_value = arg.TryToCreateOnesValue(context_, test::MakeParams(50));
  ASSERT_NE(arg_value, nullptr);
}

TEST_F(KernelArgTest, TryToCreateOnesValueGlobalIntSize) {
  cl::Kernel kernel = test::CreateClKernel("kernel void A(global int* a) {}");

  KernelArg arg;
  ASSERT_TRUE(arg.Init(&kernel, 0).ok());

  auto arg_value = arg.TryToCreateOnesValue(context_, test::MakeParams(50));
  auto array_value =
      test::Downcast<GlobalMemoryArgValue<cl_int>>(arg_value.get());
  EXPECT_EQ(array_value->size(), 50);
}

TEST_F(KernelArgTest, TryToCreateOnesValueGlobalIntValues) {
  cl::Kernel kernel = test::CreateClKernel("kernel void A(global int* a) {}");

  KernelArg arg;
  ASSERT_TRUE(arg.Init(&kernel, 0).ok());

  auto arg_value = arg.TryToCreateOnesValue(context_, test::MakeParams(50));
  auto array_value =
      test::Downcast<GlobalMemoryArgValue<cl_int>>(arg_value.get());

  for (auto value : array_value->vector()) {
    EXPECT_EQ(value, 1);
  }
}

}  // anonymous namespace
}  // namespace cldrive
}  // namespace gpu

TEST_MAIN();
