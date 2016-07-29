# (c) 2007 The Board of Trustees of the University of Illinois.

# Rules common to all makefiles

# Commands to build objects from source file using C compiler
# with gcc

# Uncomment below two lines and configure if you want to use a platform
# other than global one

#OPENCL_PATH=/scr/hskim/ati-stream-sdk-v2.3-lnx64
#OPENCL_LIB_PATH=$(OPENCL_PATH)/lib/x86_64

# gcc (default)
CC = gcc
PLATFORM_CFLAGS = 
  
CXX = g++
PLATFORM_CXXFLAGS = 
  
LINKER = g++
PLATFORM_LDFLAGS = -lm -lpthread

