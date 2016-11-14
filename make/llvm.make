# Build LLVM.
#
LlvmVersion := 3.9.0
LlvmSrc := $(src)/llvm/$(LlvmVersion)
LlvmBuild := $(build)/llvm/$(LlvmVersion)
LlvmLibDir := $(LlvmBuild)/lib
LlvmConfig := $(LlvmBuild)/bin/llvm-config
Llvm = $(LlvmConfig)

llvm: $(Llvm)
DocStrings += "llvm: build LLVM"

LlvmCMakeFlags := \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_ASSERTIONS=true \
	-DLLVM_TARGETS_TO_BUILD=X86 \
	-G Ninja -Wno-dev \
	$(NULL)

# Flags to build against LLVM + Clang toolchain
ClangLlvm_CxxFlags = \
	$(shell $(LlvmConfig) --cxxflags) \
	-isystem $(shell $(LlvmConfig) --src-root)/tools/clang/include \
	-isystem $(shell $(LlvmConfig) --obj-root)/tools/clang/include \
	-fno-rtti \
	$(NULL)

ClangLlvm_LdFlags = \
	$(shell $(LlvmConfig) --system-libs) \
	-L$(shell $(LlvmConfig) --libdir) \
	-ldl \
	-lclangTooling \
	-lclangToolingCore \
	-lclangFrontend \
	-lclangDriver \
	-lclangSerialization \
	-lclangCodeGen \
	-lclangParse \
	-lclangSema \
	-lclangStaticAnalyzerFrontend \
	-lclangStaticAnalyzerCheckers \
	-lclangStaticAnalyzerCore \
	-lclangAnalysis \
	-lclangARCMigrate \
	-lclangRewriteFrontend \
	-lclangRewrite \
	-lclangEdit \
	-lclangAST \
	-lclangLex \
	-lclangBasic \
	-lclang \
	-ldl \
	$(shell $(LlvmConfig) --libs) \
	-pthread \
	-lLLVMCppBackendCodeGen -lLLVMTarget -lLLVMMC \
	-lLLVMObject -lLLVMCore -lLLVMCppBackendInfo \
	-ldl -lcurses \
	-lLLVMSupport \
	-lcurses \
	-ldl \
	$(NULL)
# TODO: -lncurses on some systems, not -lcurses

LlvmUrlBase := http://llvm.org/releases/$(LlvmVersion)/

CachedLlvmComponents := \
	llvm \
	cfe \
	clang-tools-extra \
	compiler-rt \
	$(NULL)

LlvmTar := -$(LlvmVersion).src.tar.xz

CachedLlvmTarballs = $(addprefix $(cache)/,$(addsuffix $(LlvmTar),$(CachedLlvmComponents)))

# Fetch LLVM tarballs to local cache.
$(cache)/%$(LlvmTar):
	$(call wget,$@,$(LlvmUrlBase)$(notdir $@))

# Unpack an LLVM Tarball.
#
# Arguments:
#   $1 (str) Target directory
#   $2 (str) Source tarball
#
define unpack-llvm-tar
	$(call unpack-tar,$(LlvmSrc)/$1,$(cache)/$2$(LlvmTar),-xf)
endef

# Unpack LLVM tree from cached tarballs.
$(LlvmSrc): $(CachedLlvmTarballs)
	$(call unpack-llvm-tar,,llvm)
	$(call unpack-llvm-tar,tools/clang,cfe)
	$(call unpack-llvm-tar,tools/clang/tools/extra,clang-tools-extra)
	$(call unpack-llvm-tar,projects/compiler-rt,compiler-rt)

# Build rules.
$(Llvm): $(LlvmSrc)
	$(call print-task,BUILD,LLVM $(LlvmVersion),$(TaskMisc))
	$(V1)rm -rf $(LlvmBuild)
	$(V1)mkdir -p $(LlvmBuild)
	$(V1)cd $(LlvmBuild) && cmake $(LlvmSrc) $(LlvmCMakeFlags) >/dev/null
	$(V1)cd $(LlvmBuild) && ninja

.PHONY: distclean-llvm
distclean-llvm:
	$(V1)rm -fv -r $(LlvmSrc)
	$(V1)rm -fv -r $(LlvmBuild)
DistcleanTargets += distclean-llvm
