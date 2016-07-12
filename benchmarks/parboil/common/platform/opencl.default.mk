# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

OPENCL_PATH=foo
OPENCL_LIB_PATH=$(OPENCL_PATH)

# gcc (default)
CC = gcc
PLATFORM_CFLAGS = -include /usr/include/sys/_types.h -include /usr/include/machine/_types.h -framework OpenCL -I/usr/include/machine

CXX = g++
PLATFORM_CXXFLAGS = -include /usr/include/sys/_types.h -include /usr/include/machine/_types.h -framework OpenCL -I/usr/include/machine

LINKER = g++
PLATFORM_LDFLAGS = -lm -lpthread -framework OpenCL
