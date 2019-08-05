#include "maker/arduino/lcd_hello_world/lcd_hello_world.h"
#include "maker/arduino/interface/mock_arduino_interface.h"

#include "labm8/cpp/test.h"

namespace arduino {
namespace {

using ::testing::_;
using ::testing::AtLeast;
using ::testing::Ge;

TEST(Setup, LcdBeginIsCalled) {
  LcdHelloWorld<MockArduinoInterface> program;
  EXPECT_CALL(program.lcd(), begin(16, 2)).Times(1);
  program.Setup();
}

TEST(Loop, LcdPrintIsCalled) {
  LcdHelloWorld<MockArduinoInterface> program;
  program.Loop();
}

}  // namespace
}  // namespace arduino

TEST_MAIN();
