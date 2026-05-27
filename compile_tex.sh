#!/bin/bash
set -e

DIR="course_work"
FILE="main"

cd "$DIR"
pdflatex -interaction=nonstopmode "$FILE.tex"
biber "$FILE"
pdflatex -interaction=nonstopmode "$FILE.tex"
pdflatex -interaction=nonstopmode "$FILE.tex"
echo "Done: $FILE.pdf"
