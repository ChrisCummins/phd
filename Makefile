PDFLATEX ?= pdflatex
BIBER ?= biber
SED ?= sed
MV ?= mv

# Build the longform of the thesis by default.
all: longform

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
