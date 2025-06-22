@echo off
echo Generando archivo .bbl...
pdflatex ABC_Core_Collatz.tex
bibtex ABC_Core_Collatz
echo ¡Archivo ABC_Core_Collatz.bbl generado con éxito!
pause