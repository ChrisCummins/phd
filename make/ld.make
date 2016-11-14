#
# Linker
#
# TODO: Clang picks the linker for us, and will default to using the
# system linker. I would prefer to use LLVM's lld linker, but in
# initial tests I found that it wasn't up to the task. Perhaps with a
# later release I will give this another punt.
LD := $(CXX)

LdFlags =

%: %.o
	$(call o-link,$@,$(filter %.o,$^),\
		$($(patsubst %/,%,$@)_CxxFlags) \
		$($(patsubst %/,%,$(dir $@))_CxxFlags) \
		$($(patsubst %/,%,$@)_LdFlags) \
		$($(patsubst %/,%,$(dir $@))_LdFlags))

.PHONY: print-ld
print-ld:
	$(V2)echo $(o-link-cmd) $($(PMAKE_INVOC_DIR)_CxxFlags) \
		$($(PMAKE_INVOC_DIR)_LdFlags)
DocStrings += "print-ld: print linker invocation"
