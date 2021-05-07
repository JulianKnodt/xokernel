paper: dataflow permissions
	-@rm paper.bib
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex

dataflow:
	dot -Tpng dataflow.dot -o dataflow.png
permissions:
	dot -Tpng permissions.dot -o permissions.png
