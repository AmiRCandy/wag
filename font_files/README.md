# Figure font

The manuscript figures were rendered with Times New Roman from a file named
`font_files/times.ttf`. The font is proprietary and is not distributed here.

- Training, evaluation and reports do not need the font: use `--no-plots` with
  `python main.py run`, or the `report` command.
- To render figures, place a Times New Roman TrueType file at `font_files/times.ttf`.
  `configs/figure_integrity.json` pins the git blob hash of the file used for the
  manuscript (`51261a005cd5b17755274fd7d2bdde4edd01bf1c`); a different file is rejected
  by the figure-integrity check. To use another font, update that entry.
