#!/bin/bash
echo "▶ Compilación completa del paper"
echo "▶ Paso 1: Generar figura..."
python generate_figure.py

echo "▶ Paso 2: Generar .bbl..."
pdflatex ABC_Core_Collatz.tex
bibtex ABC_Core_Collatz

echo "▶ Paso 3: Compilar documento final..."
pdflatex ABC_Core_Collatz.tex
pdflatex ABC_Core_Collatz.tex

echo "✅ ¡Todo listo! Verifica ABC_Core_Collatz.pdf"