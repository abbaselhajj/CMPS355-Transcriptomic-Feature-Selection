#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
if command -v bibtex >/dev/null 2>&1; then
  bibtex main >/dev/null
elif command -v bibtex8 >/dev/null 2>&1; then
  bibtex8 main >/dev/null
else
  echo "No BibTeX executable found" >&2; exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
printf 'Built %s\n' "$(pwd)/main.pdf"
