PDFLATEX ?= pdflatex --shell-escape
BIBER ?= biber
SED ?= sed
MV ?= mv

# Build the longform of the thesis by default.
all: longform

longform: img
	$(SED) 's/^\\stapledtrue/% \\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex 2>&1 | tee log1.txt
	$(BIBER) thesis 2>&1 | tee log2.txt
	$(PDFLATEX) thesis.tex 2>&1 | tee log3.txt
	$(PDFLATEX) thesis.tex 2>&1 | tee log4.txt

stapled: img
	$(SED) 's/^% \\stapledtrue/\\stapledtrue/' -i thesis.tex
	$(PDFLATEX) thesis.tex
	$(BIBER) thesis
	$(PDFLATEX) thesis.tex
	$(PDFLATEX) thesis.tex
	$(MV) thesis.pdf thesis-stapled.pdf

# Do some PDF transcoding magic to fix weird warnings from LaTeX. I neither
# know or care to know what the correct approach is... this works.
# <https://tex.stackexchange.com/a/78009>
# <https://tex.stackexchange.com/a/150745>
img:
	for img in $$(ls img/*.pdf); do \
		gs -o tmp.pdf -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 \
				-dColorConversionStrategy=/sRGB \
				-dProcessColorModel=/DeviceRGB $$img && \
		pdf2ps tmp.pdf tmp2.pdf && \
		ps2pdf tmp2.pdf tmp.pdf && \
		mv -v tmp.pdf $$img; \
	done
	rm tmp2.pdf

help:
	@echo "make {all|longform|stapled}"

.PHONY: all longform stapled img help
