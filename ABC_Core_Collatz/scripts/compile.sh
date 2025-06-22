#!/bin/bash  
echo "Compilando paper LaTeX..."  
pdflatex ABC_Core_Collatz.tex  
bibtex ABC_Core_Collatz  
pdflatex ABC_Core_Collatz.tex  
pdflatex ABC_Core_Collatz.tex  
echo "✅ ¡Compilación exitosa! Verifica ABC_Core_Collatz.pdf"  
# Ejecuta esto para verificar errores
chktex ABC_Core_Collatz.tex
bibexport -o extracted.bib ABC_Core_Collatz