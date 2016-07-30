#
# An implementation of the feature extractor used in:
#
#     Grewe, D., Wang, Z., & O’Boyle, M. F. P. M. (2013). Portable
#     mapping of data parallel programs to OpenCL for heterogeneous
#     systems. In CGO. IEEE.
#
# Written by Zheng Wang <zh.wang@ed.ac.uk>.
# Modified by Chris Cummins <chrisc.101@gmail.com>
#
CXX := $(HOME)/phd/tools/llvm/build/bin/clang++
LLVMCOMPONENTS :=
RTTIFLAG := -fno-rtti
LLVMCONFIG := $(HOME)/phd/tools/llvm/build/bin/llvm-config

CXXFLAGS := -I$(shell $(LLVMCONFIG) --src-root)/tools/clang/include \
	-I$(shell $(LLVMCONFIG) --obj-root)/tools/clang/include \
	$(shell $(LLVMCONFIG) --cxxflags) $(RTTIFLAG) -Iinclude -g
LLVMLDFLAGS := $(shell $(LLVMCONFIG) --ldflags --libs)

SOURCES = FeatureExtractor.cpp

HEADERS = include/FeatureExtractor.h

EXECUTABLE = featureExtractor
OBJECTS = $(SOURCES:.cpp=.o)
CLANGLIBS = \
				-lclangTooling\
				-lclangFrontendTool\
				-lclangFrontend\
				-lclangDriver\
				-lclangSerialization\
				-lclangCodeGen\
				-lclangParse\
				-lclangSema\
				-lclangStaticAnalyzerFrontend\
				-lclangStaticAnalyzerCheckers\
				-lclangStaticAnalyzerCore\
				-lclangAnalysis\
				-lclangARCMigrate\
				-lclangRewrite\
				-lclangRewriteFrontend\
				-lclangEdit\
				-lclangAST\
				-lclangLex\
				-lclangBasic\
				$(shell $(LLVMCONFIG) --libs)\
				$(shell $(LLVMCONFIG) --system-libs)\
                -lcurses

all: $(EXECUTABLE) $(SOURCES) $(HEADERS)


$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) $(CLANGLIBS) $(LLVMLDFLAGS)  -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<


clean:
	-rm -f $(EXES) $(OBJECTS) *~
