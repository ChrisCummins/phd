PDFLATEX ?= pdflatex
BIBER ?= biber
SED ?= sed
MV ?= mv

# First build the 'stapled' version of the thesis, then the normal version.
all: stapled longform

longform:
	$(SED) 's/^\\stapledtrue/% \\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex
	$(BIBER) thesis
	$(PDFLATEX) thesis.tex
	$(PDFLATEX) thesis.tex

stapled:
	$(SED) 's/^% \\stapledtrue/\\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex
	$(BIBER) thesis
	$(PDFLATEX) thesis.tex
	$(PDFLATEX) thesis.tex
	$(MV) thesis.pdf thesis-stapled.pdf

help:
	@echo "make {all|longform|stapled}"

.PHONY: all longform stapled help
