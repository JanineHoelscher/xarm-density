# Build the UF850 paper

Tracked files include both LaTeX sources, the seven figure PDFs in
`../figures/`, recorded CSV/JSON inputs in `../figures/data/`, and numerical
plotting helpers. The original archive is not needed for these workflows.
The published [root PDF](../../density-xarm.pdf) remains unchanged.

## Compile from a fresh clone

Use TeX Live with IEEEtran and the packages imported at the top of the document.
No conda/ROS installation is needed just to compile. This host provides system
`latexmk` and `pdflatex` with IEEEtran. From the repository root:

```bash
cd scripts/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build density-xarm-optimized.tex
```

Output: `scripts/latex/build/density-xarm-optimized.pdf`. Auxiliary files and
compiled PDFs in this folder are gitignored. To build the earlier layout,
replace the source filename with `density-xarm.tex`; only its figure lookup
was changed during restoration. All paths are relative to the clone.

The optimized source is the recommended compact IEEE layout. It previously
produced nine pages; pagination can vary with the TeX distribution. Scientific
content, IEEEtran margins, vector plots, and the tight 1x4 snapshot panel are
preserved. Changes concern floats, microtypography, highlight boxes, and columns.

## Replot recorded results

From the repository root:

```bash
conda activate xarm
python -m pip install -e '.[paper]'
python scripts/figures/generate_figures.py --from-csv
```

This validates tracked CSV/JSON checksums, reconstructs temporary plotting
inputs, and regenerates figures 1--4 and 6--7 and their tables. Figure 8 is
the included recorded snapshot panel, not a new render. See
[figure reproduction](../figures/README.md) for details.

Replotting preserves recorded measurements; it does not rerun historical
controllers or collect new timing measurements. New simulations use the
current standalone runner documented in the main README.

## Overleaf

Use `density-xarm-optimized.tex` as the main document with pdfLaTeX. Include
the seven figure PDFs in `figures/`, or retain the sibling `latex/` and
`figures/` folders. Both layouts are supported. The bibliography is inline;
no separate `.bib` or custom class file is required. IEEEtran must be available
in the chosen TeX installation.

## Verification

Both layouts compiled successfully after restoration; the optimized build has
nine pages. CSV restoration and both numerical plotters ran without archived
inputs. All four regenerated CSV tables matched the preserved tables
byte-for-byte. Figure exports may differ in PDF metadata or rendering details.
