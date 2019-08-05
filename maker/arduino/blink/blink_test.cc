#include "maker/arduino/blink/blink.h"
#include "maker/arduino/interface/mock_arduino_interface.h"

#include "labm8/cpp/test.h"

namespace arduino {
namespace blink {
namespace {

using ::testing::_;
using ::testing::AtLeast;

TEST(Setup, PinModeIsCalled) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface(),
              SetPinMode(MockArduinoInterface::kBuiltInLedPin, _))
      .Times(1);
  program.Setup();
}

TEST(Loop, DigitalWriteIsCalled) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface(),
              DigitalWrite(MockArduinoInterface::kBuiltInLedPin, _))
      .Times(AtLeast(2));
  program.Loop();
}

TEST(Loop, DelayIsCalled) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface(), Delay(_)).Times(AtLeast(2));
  program.Loop();
}

}  // namespace
}  // namespace blink
}  // namespace arduino

TEST_MAIN();
