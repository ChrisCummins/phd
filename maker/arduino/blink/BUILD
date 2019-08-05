# "Hello world" for microcontrollers. Flash an LED on and off.

load(
    "@platformio_rules//platformio:platformio.bzl",
    "platformio_library",
    "platformio_project",
)

# The implementation of the Setup() and Loop() functions.
cc_library(
    name = "blink",
    hdrs = ["blink.h"],
    deps = ["//maker/arduino/interface:arduino_interface"],
)

# A C++ test of blink logic using mock Arduino interface.
cc_test(
    name = "blink_test",
    srcs = ["blink_test.cc"],
    deps = [
        ":blink",
        "//labm8/cpp:test",
        "//maker/arduino/interface:mock_arduino_interface",
    ],
)

# The blink implementation as a library for PlatformIO projects.
platformio_library(
    name = "Blink",
    hdr = "blink.h",
)

platformio_project(
    name = "arduino_uno",
    src = "arduino_uno.cc",
    board = "uno",
    # If you need to you can pass multiple string key-value pairs directly
    # to PlatformIO. All of these will appear under the environment section in
    # the generated platformio.ini file. For the available options see:
    # http://docs.platformio.org/en/stable/projectconf.html
    environment_kwargs = {
        "board_build.f_cpu": "16000000L",
    },
    framework = "arduino",
    platform = "atmelavr",
    deps = [
        ":Blink",
        "//maker/arduino/interface:Arduino_hardware",
        "//maker/arduino/interface:Arduino_interface",
    ],
)
