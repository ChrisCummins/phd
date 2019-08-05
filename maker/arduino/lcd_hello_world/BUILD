# "Hello world" for microcontrollers. Flash an LED on and off.

load(
    "@platformio_rules//platformio:platformio.bzl",
    "platformio_library",
    "platformio_project",
)

# The implementation of the Setup() and Loop() functions.
cc_library(
    name = "lcd_hello_world",
    hdrs = ["lcd_hello_world.h"],
    deps = ["//maker/arduino/interface:arduino_interface"],
)

# A C++ test of lcd_hello_world logic using mock Arduino interface.
# TODO(cec): Re-enable after LiquidCrystalInterface has been implemented.
#cc_test(
#    name = "lcd_hello_world_test",
#    srcs = ["lcd_hello_world_test.cc"],
#    deps = [
#        ":lcd_hello_world",
#        "//maker/arduino/interface:mock_arduino_interface",
#        "//labm8/cpp:test",
#    ],
#)

# The lcd_hello_world implementation as a library for PlatformIO projects.
platformio_library(
    name = "LcdHelloWorld",
    hdr = "lcd_hello_world.h",
)

platformio_project(
    name = "arduino_uno",
    src = "arduino_uno.cc",
    board = "uno",
    framework = "arduino",
    platform = "atmelavr",
    deps = [
        ":LcdHelloWorld",
        "//maker/arduino/interface:Arduino_hardware",
        "//maker/arduino/interface:Arduino_interface",
        "//third_party/arduino/LiquidCrystal",
    ],
)
