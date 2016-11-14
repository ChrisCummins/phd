#
# LaTeX
#
BuildTargets += $(AutotexTargets)

AutotexDirs = $(dir $(AutotexTargets))
AutotexDepFiles = $(addsuffix .autotex.deps, $(AutotexDirs))
AutotexLogFiles = $(addsuffix .autotex.log, $(AutotexDirs))

# Autotex does it's own dependency analysis, so always run it:
.PHONY: $(AutotexTargets)
$(AutotexTargets):
	$(V2)$(root)/make_tools/autotex.sh make $(patsubst %.pdf,%,$@)

# File extensions to remove in LaTeX build directories:
LatexBuildfileExtensions = \
	-blx.bib \
	.acn \
	.acr \
	.alg \
	.aux \
	.bbl \
	.bcf \
	.blg \
	.dvi \
	.fdb_latexmk \
	.glg \
	.glo \
	.gls \
	.idx \
	.ilg \
	.ind \
	.ist \
	.lof \
	.log \
	.lol \
	.lot \
	.maf \
	.mtc \
	.mtc0 \
	.nav \
	.nlo \
	.out \
	.pdfsync \
	.ps \
	.run.xml \
	.snm \
	.synctex.gz \
	.tdo \
	.toc \
	.vrb \
	.xdy \
	$(NULL)

LatexBuildDirs = $(AutotexDirs)

# Discover files to remove using the shell's `find' tool:
LatexCleanFiles = \
	$(shell find $(LatexBuildDirs) -name '*$(call join-with,' -o -name '*, $(LatexBuildfileExtensions))') \
	$(AutotexTargets) \
	$(AutotexDepFiles) \
	$(AutotexLogFiles) \
	$(shell biber --cache) \
	$(NULL)
CleanFiles += $(LatexCleanFiles)

.PHONY: clean-tex
clean-tex:
	$(V1)rm -rfv $(sort $(LatexCleanFiles))
DocStrings += "clean-tex: remove generated LaTeX files"

tex: $(AutotexTargets)
DocStrings += "tex: build all LaTeX targets"
