# Build the working report and the paper.
#
#   make            build report.pdf and paper.pdf
#   make report     build report.pdf (working report, drafted with Claude Code)
#   make paper      build paper.pdf
#   make figures    regenerate figures/ and tables from the results in ../runs
#   make clean      remove build by-products
#   make distclean  also remove the PDFs
#
# Uses tectonic if available, else latexmk, else pdflatex + bibtex.

DOCS    := report paper
AUX     := $(foreach d,$(DOCS),$(addprefix $(d).,aux bbl blg log out toc fls fdb_latexmk synctex.gz))
DEPS    := references.bib $(wildcard figures/*.pdf figures/*.tex)

TECTONIC := $(shell command -v tectonic 2>/dev/null)
LATEXMK  := $(shell command -v latexmk 2>/dev/null)

.PHONY: all report paper figures clean distclean

all: report.pdf paper.pdf

report: report.pdf

paper: paper.pdf

%.pdf: %.tex $(DEPS)
ifneq ($(TECTONIC),)
	tectonic -X compile $<
else ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<
else
	pdflatex -interaction=nonstopmode -halt-on-error $<
	bibtex $*
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<
endif

figures:
	cd .. && uv run imonet figures --out paper/figures

clean:
	rm -f $(AUX)

distclean: clean
	rm -f $(addsuffix .pdf,$(DOCS))
