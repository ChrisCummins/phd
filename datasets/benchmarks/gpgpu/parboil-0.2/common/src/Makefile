# (c) 2007 The Board of Trustees of the University of Illinois.

SRCDIR=$(PARBOIL_ROOT)/common/src
BUILD_DIR=$(PARBOIL_ROOT)/common/lib

#include $(PARBOIL_ROOT)/common/mk/common.mk
#include $(PARBOIL_ROOT)/common/mk/cuda.mk

include $(PARBOIL_ROOT)/common/Makefile.conf

CFLAGS=-I$(PARBOIL_ROOT)/common/include

AR=ar
RANLIB=ranlib

OBJS=$(BUILD_DIR)/parboil.o
OBJS_TWO=$(BUILD_DIR)/parboil_cuda.o
OBJS_THREE=$(BUILD_DIR)/parboil_opencl.o
TARGET=$(BUILD_DIR)/libparboil.a 
TARGET_TWO=$(BUILD_DIR)/libparboil_cuda.a
TARGET_THREE=$(BUILD_DIR)/libparboil_opencl.a

ifeq ($(CUDA_PATH),)
TARGET_CUDA=no_cuda
else
TARGET_CUDA=$(BUILD_DIR)/libparboil_cuda.a
endif

ifeq ($(OPENCL_PATH),)
TARGET_OPENCL=no_opencl
else
TARGET_OPENCL=$(BUILD_DIR)/libparboil_opencl.a
endif

TARGETS=$(TARGET) $(TARGET_CUDA) $(TARGET_OPENCL)

.PHONY: all clean

all : $(TARGETS)

no_cuda:
	@echo
	@echo CUDA_PATH is not set in $$PARBOIL_ROOT/common/Makefile.conf. CUDA ignored. | grep --color CUDA_PATH
	@echo

no_opencl:
	@echo
	@echo OPENCL_PATH is not set in $$PARBOIL_ROOT/common/Makefile.conf. OPENCL ignored. | grep --color OPENCL_PATH
	@echo

clean:
	rm -f $(OBJS) $(TARGETS)

$(BUILD_DIR)/parboil.o : parboil.c
	$(CC) -c -g $(CFLAGS) $^ -o $@

$(BUILD_DIR)/parboil_cuda.o : parboil_cuda.c
	$(CC) -c -g $(CFLAGS) -I$(CUDA_PATH)/include $^ -o $@

$(BUILD_DIR)/parboil_opencl.o : parboil_opencl.c
	$(CC) -c -g $(CFLAGS) -I$(OPENCL_PATH)/include $^ -o $@

$(TARGET) : $(OBJS)
	$(AR) rc $@ $?
	$(RANLIB) $@

$(TARGET_TWO) : $(OBJS_TWO)
	$(AR) rc $@ $?
	$(RANLIB) $@
	
$(TARGET_THREE) : $(OBJS_THREE)
	$(AR) rc $@ $?
	$(RANLIB) $@	
