#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "phd/port.h"
#include "phd/test.h"

namespace gpu {
namespace cldrive {
namespace {

TEST(ScalarKernelArgValue, IntValuesAreEqual) {
  ScalarKernelArgValue<phd::int32> a(5);
  ScalarKernelArgValue<phd::int32> b(5);
  EXPECT_EQ(a, &b);
}

TEST(ScalarKernelArgValue, DifferentIntValuesAreNotEqual) {
  ScalarKernelArgValue<phd::int32> a(5);
  ScalarKernelArgValue<phd::int32> b(6);
  EXPECT_NE(a, &b);
}

TEST(ScalarKernelArgValue, FloatValuesAreEqual) {
  ScalarKernelArgValue<float> a(1.25);
  ScalarKernelArgValue<float> b(1.25);
  EXPECT_EQ(a, &b);
}

TEST(ScalarKernelArgValue, DifferentFloatValuesAreNotEqual) {
  ScalarKernelArgValue<float> a(1.25);
  ScalarKernelArgValue<float> b(1.35);
  EXPECT_NE(a, &b);
}

TEST(ScalarKernelArgValue, DifferentTypesWithSameValueAreNotEqual) {
  ScalarKernelArgValue<phd::int32> a(5);
  ScalarKernelArgValue<phd::int64> b(5);
  EXPECT_NE(a, &b);
}

TEST(ScalarKernelArgValue, IntValueToString) {
  EXPECT_EQ(ScalarKernelArgValue<phd::int32>(3).ToString(), string("3"));
  EXPECT_EQ(ScalarKernelArgValue<phd::int64>(3).ToString(), string("3"));
}

TEST(ScalarKernelArgValue, FloatValueToString) {
  EXPECT_EQ(ScalarKernelArgValue<float>(3.5).ToString(), string("3.5"));
  EXPECT_EQ(ScalarKernelArgValue<float>(0.12345).ToString(), string("0.12345"));
}

}  // anonymous namespace
}  // namespace cldrive
}  // namespace gpu

TEST_MAIN();
