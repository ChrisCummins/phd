PDFLATEX ?= pdflatex
BIBER ?= biber
SED ?= sed
MV ?= mv

# First build the 'stapled' version of the thesis, then the normal version.
all:
	$(SED) 's/^% \\stapledtrue/\\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex
	$(BIBER) thesis
	$(PDFLATEX) thesis.tex
	$(PDFLATEX) thesis.tex
	$(MV) thesis.pdf thesis-stapled.pdf
	$(SED) 's/^\\stapledtrue/% \\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex
	$(BIBER) thesis
	$(PDFLATEX) thesis.tex
	$(PDFLATEX) thesis.tex

.PHONY: all
