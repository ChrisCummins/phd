GoogleTestVersion := 1.8.0
GoogleTestSrc := $(src)/googletest/$(GoogleTestVersion)
GoogleTestBuild := $(build)/googletest/$(GoogleTestVersion)
GoogleTest := $(GoogleTestBuild)/libgtest.a

googletest: $(GoogleTest)
DocStrings += "googletest: build Google Test library"

GoogleTest_CxxFlags = -isystem $(GoogleTestSrc)/googletest/include
GoogleTest_LdFlags = -lpthread -L$(GoogleTestBuild) -lgtest

CachedGoogleTestTarball = $(cache)/googletest-$(GoogleTestVersion).tar.gz
GoogleTestUrl = https://github.com/google/googletest/archive/release-$(GoogleTestVersion).tar.gz

# Download tarball
$(CachedGoogleTestTarball):
	$(call wget,$(CachedGoogleTestTarball),$(GoogleTestUrl))

# Unpack tarball
$(GoogleTestSrc): $(CachedGoogleTestTarball)
	$(call unpack-tar,$(GoogleTestSrc),$<,-zxf)

# Build
$(GoogleTest)-cmd = \
	cd $(GoogleTestBuild) \
	&& cmake $(GoogleTestSrc)/googletest -G Ninja >/dev/null \
	&& ninja

$(GoogleTest): $(GoogleTestSrc)
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)rm -rf $(GoogleTestBuild)
	$(V1)mkdir -p $(GoogleTestBuild)
	$(V1)$($(GoogleTest)-cmd)

# Clean
.PHONY: distclean-googletest
distclean-googletest:
	$(V1)rm -fv -r $(GoogleTestSrc)
	$(V1)rm -fv -r $(GoogleTestBuild)
DistcleanTargets += distclean-googletest