GoogleBenchmarkVersion := 1.0.0
GoogleBenchmarkSrc := $(src)/googlebenchmark/$(GoogleBenchmarkVersion)
GoogleBenchmarkBuild := $(build)/googlebenchmark/$(GoogleBenchmarkVersion)
GoogleBenchmark := $(GoogleBenchmarkBuild)/src/libbenchmark.a

googlebenchmark: $(GoogleBenchmark)
DocStrings += "googlebenchmark: build Google Benchmark library"

GoogleBenchmark_CxxFlags = -isystem $(GoogleBenchmarkSrc)/include \
	-Wno-global-constructors
GoogleBenchmark_LdFlags = -lpthread -L$(GoogleBenchmarkBuild)/src -lbenchmark

CachedGoogleBenchmarkTarball = $(cache)/googlebenchmark-$(GoogleBenchmarkVersion).tar.gz
GoogleBenchmarkUrl = https://github.com/google/benchmark/archive/v$(GoogleBenchmarkVersion).tar.gz

# Download tarball
$(CachedGoogleBenchmarkTarball):
	$(call wget,$(CachedGoogleBenchmarkTarball),$(GoogleBenchmarkUrl))

# Unpack tarball
$(GoogleBenchmarkSrc): $(CachedGoogleBenchmarkTarball)
	$(call unpack-tar,$(GoogleBenchmarkSrc),$<,-zxf)

# Build
$(GoogleBenchmark)-cmd = \
	cd $(GoogleBenchmarkBuild) \
	&& cmake $(GoogleBenchmarkSrc) -G Ninja >/dev/null \
	&& ninja

$(GoogleBenchmark): $(GoogleBenchmarkSrc)
	$(call print-task,BUILD,$@,$(TaskMisc))
	$(V1)rm -rf $(GoogleBenchmarkBuild)
	$(V1)mkdir -p $(GoogleBenchmarkBuild)
	$(V1)$($(GoogleBenchmark)-cmd)

# Clean
.PHONY: distclean-googlebenchmark
distclean-googlebenchmark:
	$(V1)rm -fv -r $(GoogleBenchmarkSrc)
	$(V1)rm -fv -r $(GoogleBenchmarkBuild)
DistcleanTargets += distclean-googlebenchmark