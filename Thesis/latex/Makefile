
default:
	latexmk -pdfdvi thesis.tex

cont:
	#latexmk -pvc -view=dvi thesis.tex
	latexmk -pdfdvi -pvc -view=pdf thesis.tex

clean:
	@latexmk -C
	@rm -f *.brf
	@rm -f *.lol
	@rm -f *.out
	
