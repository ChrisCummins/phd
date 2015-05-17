# Import common build script.
include $(top_srcdir)/Makefile.common

AM_CXXFLAGS = -isystem $(OPENCL_HEADERS) \
	`pkg-config --cflags glibmm-2.4 giomm-2.4`
AM_LDFLAGS = -L$(OPENCL_LIB) -lOpenCL \
	`pkg-config --libs glibmm-2.4 giomm-2.4`

noinst_PROGRAMS = 			\
	client				\
	$(NULL)

client_SOURCES = client.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
