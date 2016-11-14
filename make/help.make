# Print information. 'make help' helper.
define print-info
	echo $1 | xargs printf "    %-10s $2\n"
endef

print-program-version-cmd = $(shell which $1 &>/dev/null && \
	{ $1 --version 2>&1 | head -n1; } || { echo not found; })

define print-program-version
	$(call print-info,$1,$(print-program-version-cmd))
endef

# Print doc strings:
.PHONY: help
help:
	$(V2)echo "usage: make [argument...] [target...]"
	$(V2)echo
	$(V2)echo "values for arguments:"
	$(V2)echo
	$(V2)(for var in $(ArgStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-20s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
	$(V2)echo "values for targets (default=all):"
	$(V2)echo
	$(V2)(for var in $(DocStrings); do echo $$var; done) \
		| sort --ignore-case | while read var; do \
		echo $$var | cut -f 1 -d':' | xargs printf "    %-20s "; \
		echo $$var | cut -d':' -f2-; \
	done
	$(V2)echo
	$(V2)echo "host info:"
	$(V2)echo
	$(V2)$(call print-info,name,$(shell uname -n))
	$(V2)$(call print-info,O/S,$(shell uname -o))
	$(V2)$(call print-info,arch,$(shell uname -m))
	$(V2)$(call print-info,threads,$(threads))
	$(V2)echo
	$(V2)echo "build essentials:"
	$(V2)echo
	$(V2)$(call print-program-version,c++)
	$(V2)$(call print-program-version,cmake)
	$(V2)$(call print-program-version,ninja)
	$(V2)$(call print-program-version,pdflatex)
	$(V2)$(call print-program-version,pep8)
	$(V2)$(call print-program-version,python2)
	$(V2)$(call print-program-version,python3)
	$(V2)$(call print-program-version,svn)
	$(V2)echo
