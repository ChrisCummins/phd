# Top level makefile

# Input to this Makefile
# $(PATH_TO_APP)/Makefile
# $(PARBOIL_ROOT)
# $(PLATFORM)

ifeq ($(PLATFORM),)
PLATFORM=default
endif

include $(PARBOIL_ROOT)/common/Makefile.conf

SRC_MK=$(SRCDIR)/Makefile
include $(SRC_MK)

PLATFORM_MK=$(PARBOIL_ROOT)/common/platform/$(LANGUAGE).$(PLATFORM).mk
include $(PLATFORM_MK)

LANGUAGE_MK=$(PARBOIL_ROOT)/common/mk/$(LANGUAGE).mk
include $(LANGUAGE_MK)

