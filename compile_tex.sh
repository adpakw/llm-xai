#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")/course_work" && pwd)"
FILE="main"

cd "$DIR"
pdflatex -interaction=nonstopmode "$FILE.tex"
biber "$FILE"
pdflatex -interaction=nonstopmode "$FILE.tex"
pdflatex -interaction=nonstopmode "$FILE.tex"
echo "Done: $DIR/$FILE.pdf"
