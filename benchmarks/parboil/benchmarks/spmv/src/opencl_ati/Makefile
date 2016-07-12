# (c) 2010 The Board of Trustees of the University of Illinois.


LANGUAGE=opencl
TOOLS_SRC=common_src/convert-dataset
SRCDIR_OBJS=main.o gpu_info.o file.o ocl.o 
APP_CUDALDFLAGS=-lm
APP_CFLAGS=-ffast-math -g3 -O3 -I$(TOOLS_SRC)
APP_CXXFLAGS=-ffast-math -g3 -O3

include $(TOOLS_SRC)/commontools.mk

