# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
SRCDIR_OBJS=main.o file.o
APP_LDFLAGS= -lm -lstdc++

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	EXTRA_CFLAGS += -fp-model precise
else
	EXTRA_CFLAGS += -ffast-math
	EXTRA_CXXFLAGS += -ffast-math
	EXTRA_CUDACFLAGS += --use_fast_math
endif

EXTRA_LIBS += -lm -lstdc++

# To use the MKL library
ifeq ($(mkl), 1)
	EXTRA_LIBS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	INCLUDES += -I/opt/intel/mkl/9.1.023/include
	EXTRA_CFLAGS += -DMKL
endif
