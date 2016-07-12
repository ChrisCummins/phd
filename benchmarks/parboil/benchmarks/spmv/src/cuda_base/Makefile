# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
TOOLS_SRC=common_src/convert-dataset
SRCDIR_OBJS=main.o file.o gpu_info.o 
APP_LDFLAGS= -lm -lstdc++
APP_CFLAGS=-I$(TOOLS_SRC)
APP_CUDACFLAGS=-I$(TOOLS_SRC)

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	EXTRA_CFLAGS += -fp-model precise
else
	EXTRA_CFLAGS += -ffast-math
	EXTRA_CXXFLAGS += -ffast-math
	EXTRA_CUDACFLAGS += --use_fast_math
endif

# To use the MKL library
ifeq ($(mkl), 1)
	EXTRA_LIBS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	INCLUDES += -I/opt/intel/mkl/9.1.023/include
	EXTRA_CFLAGS += -DMKL
endif

include $(TOOLS_SRC)/commontools.mk
