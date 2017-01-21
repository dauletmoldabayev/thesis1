
Introduction
============

This is a LaTeX template for a research-paper-based PhD dissertation at the
University of Bergen. The generated pdf follows the standards set by UiB, and
can be sent directly to AIT for printing (That is at least how the authors have
produced their dissertations). 


Prerequisites
=============

You will need the following programs
1) pdffont:       Check if fonts are embedded in a pdf file
2) pdftk:         PDF toolkit - manipulates pdf files: rotate, burst, merge, etc. 
3) pdfcrop:       Removes whitespace from pdf files
4) latexmk:       perl script for compiling latex files (uses latex and dvipdf*)
5) pdftops:       Converting pdf files to ps using the poppler library
6) ps2pdf:        Converting ps files to pdf using ghostscript
7) dvipdfmx:      An improved version of dvipdf, converting from dvi to pdf
8) sam2p:         Another program for converting ps to pdf

Optional (but useful) programs:
9) image magic:   Edit, convert and compose images
10) TeXcount: word statistics of .tex document (http://folk.uio.no/einarro/Comp/texwordcount.html)
11) latexdiff: formatted diff of two latex documents (http://www.ctan.org/tex-archive/support/latexdiff)

You will also need a host of latex packages, as well as the latex compiler. The
easiest way of obtaining all this on ubuntu is to install the texlive-full
package. 

To install all required programs and packages in ubuntu (at least for version
9.04), run the following command:

> sudo apt-get install texlive-full poppler-utils pdftk latexmk dvipdfmx sam2p imagemagick


Usage
=====

Generating ps files of the papers from pdfs: 
--------------------------------------------

In order to create a single pdf file containing the dissertation with all
research articles (some of which may be available in pdf format typeset by the
publisher), all papers should be placed in the "papers" folder with the naming
convention 

"paper/paper[i].pdf" 

where [i] is replaced with the index of the paper in the paper list (with 1 as
base). A dummy-paper (paper1.pdf) is included for reference.

In order to include the pdfs in the dissertation, they must first be split up
in single pages and stored as .ps-files. A tool is included, which performs
this action. It can be invoked from the papers folder by running the command

> ./prep_papers_thesisinclude.py --paper-count=[number of papers] --prepare-all

If successful, this should generate the folders "paper/paper-1/" through
"paper/paper-n/" where n is the number of papers.

Finally, to include the pages from these papers, you must edit
"latex/papers.tex" and modify paper title, authors, publication as well as
number of pages.

Compiling latex source to pdf
-----------------------------

Having bursted created the paper pages as ps-files, the dissertation pdf file
can now be created. Enter the latex folder and run the command 

> make

if successful, thesis.pdf should now be generated.


Some PDF Tricks
===============

Force font embedding while converting from PS to PDF (version 1.4):
> ps2pdf14 -dPDFSETTINGS=/prepress -dEmbedAllFonts=true avhandling.ps

Install non-free fonts on ubuntu:
> getnonfreefonts-sys --all

Update font maps
> sudo updmap-sys

Include all fonts with ghostscript:
> latex myfile.tex
> dvips myfile
> gs -dSAFER -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sPAPERSIZE=a4 -dPDFSETTINGS=/printer -dCompatibilityLevel=1.3 -dMaxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sOutputFile=file.pdf myfile.ps 

Scale a PS-file to 80% of original size:
> dvips -x 800 -o myfile.ps myfile.dvi

"Converting bitmap images to EPS can be done with either sam2p or image magick's convert
> sam2p myimg.png EPS: myimg.eps
> convert myimg.png eps3:myimg.eps

Classify LaTeX symbols (draw -> get LaTeX symbol name)
http://detexify.kirelabs.org/classify.html


License
=======

This work is licensed under a Creative Commons Attribution 3.0 Unported License
See http://creativecommons.org/licenses/by/3.0/ for more details

You can basically use it for whatever you want as long as you attribute the
original authors


Authors
=======

Raymond Nepstad
Tore Birkeland
