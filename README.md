# imonet

Virtual GNSS stations: daily residual position series generated at arbitrary
locations from the surrounding stations with a conditional diffusion model,
trained and validated on California and tested on Türkiye. The design is in
`docs/00-design.md`; the report and paper are in `paper/`.

## Data

Everything is fetched and built into `data/` (not tracked), with no other
project needed:

```
uv run imonet fetch meta                      # NGL station table and step catalogue
uv run imonet fetch series --region california
uv run imonet fetch series --region turkiye
uv run imonet fetch cfm                       # SCEC Community Fault Model (experiment B)
uv run imonet build --region california       # data/cube/california.npz
uv run imonet build --region turkiye          # data/cube/turkiye.npz
```

The California download is about 1.6 GB from NGL's server and takes hours.
`fetch meta` keeps a cached copy; the cubes used so far were built with the
station table and step catalogue of 2026-10-06 (a newer catalogue can list
new steps and change the cube).
Results of runs go to `runs/` (not tracked).

When a scoring rule changes, `imonet rescore` (experiment A) and `imonet
rescore --experiment B` recompute the stored summaries without new samples;
the classifier test filters its cached windows when rerun. The tables the
documents `\input` then follow from `imonet figures`, but numbers quoted in
the prose of `paper/report.tex`, `notes/technical-notes.tex` and
`docs/00-design.md` do not: grep them for the old headline values before
committing a rescore.

## License

The code is released under the MIT License (`LICENSE`). The paper, report and
figures in `paper/` are released under the Creative Commons Attribution 4.0
International License (`paper/LICENSE`).
