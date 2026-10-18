# Build the technical notes.
#
#   make            build technical-notes.pdf
#   make tables     regenerate the result tables it includes (../paper/figures)
#   make clean      remove build by-products
#   make distclean  also remove the PDF
#
# Uses tectonic if available, else latexmk, else pdflatex (twice, for the
# table of contents).

DOC  := technical-notes
DEPS := $(wildcard ../paper/figures/*.tex)

TECTONIC := $(shell command -v tectonic 2>/dev/null)
LATEXMK  := $(shell command -v latexmk 2>/dev/null)

.PHONY: all tables clean distclean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex $(DEPS)
ifneq ($(TECTONIC),)
	tectonic -X compile $<
else ifneq ($(LATEXMK),)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $<
else
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<
endif

tables:
	cd .. && uv run imonet figures --out paper/figures

clean:
	rm -f $(addprefix $(DOC).,aux log out toc fls fdb_latexmk synctex.gz)

distclean: clean
	rm -f $(DOC).pdf
