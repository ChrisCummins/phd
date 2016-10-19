# Build LLVM.
#
Llvm = $(build)/llvm/3.9.0/bin/llvm-config

DocStrings += "llvm: build LLVM"
DocStrings += "clean-llvm: remove LLVM build"
DocStrings += "distclean-llvm: remove build and source"

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

.PHONY: clean-llvm distclean-llvm

clean-llvm:
	$(V1)rm -fv -r $(LlvmBuild)

distclean-llvm: clean-llvm
	$(V1)rm -fv -r $(LlvmSrc)
