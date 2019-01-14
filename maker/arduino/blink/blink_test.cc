#include "maker/arduino/blink/blink.h"
#include "maker/arduino/interface/mock_arduino_interface.h"

#include "phd/test.h"

namespace arduino {
namespace blink {
namespace {

using ::testing::_;

TEST(Setup, PinModeIsCalled) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface_, SetPinMode(kLedToFlash, _)).Times(1);
  program.Setup();
}

TEST(Loop, DigitalWriteIsCalledTwice) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface_, DigitalWrite(kLedToFlash, _)).Times(2);
  program.Loop();
}

TEST(Loop, DelayIsCalledTwice) {
  Blink<MockArduinoInterface> program;
  EXPECT_CALL(program.interface_, Delay(_)).Times(2);
  program.Loop();
}

}  // namespace
}  // namespace blink
}  // namespace arduino

TEST_MAIN();
