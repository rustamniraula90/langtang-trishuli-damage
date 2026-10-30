# Where the Flood Reached: terrain-only classical machine learning of building damage in the 2026 Langtang Lirung–Trishuli debris flood, Nepal

Code, derived tables, label-audit record and manuscript source for the paper of the same name
(Rustam Niraula, MSc Data Science and Computational Intelligence, Softwarica College / Coventry University, 2026).

The study joins Copernicus EMSR927 building-damage grades to pre-event Overture footprints (17,503 buildings in three
reaches of the Bhote Koshi–Trishuli valley) and derives ten terrain features from the Copernicus GLO-30 DEM. Nine
classical learners are tested with leave-one-reach-out evaluation and 1-km block bootstraps.

Release v1.0.1 is archived on Zenodo: [10.5281/zenodo.23802509](https://doi.org/10.5281/zenodo.23802509)
(concept DOI [10.5281/zenodo.23802508](https://doi.org/10.5281/zenodo.23802508) always resolves to the latest release).
Citation details are in `CITATION.cff`.

## Layout

| Path | Content |
|---|---|
| `pilot/*.py`, `pilot/run_all.sh` | The pipeline, steps 01–30. `run_all.sh` rebuilds every number and figure in order. |
| `pilot/requirements.txt` | Pinned Python environment (Python 3.12.16) used for every result in the paper. |
| `pilot/README.md` | Step-by-step notes on the pipeline (pilot record; the paper and `results/*.csv` are authoritative where numbers differ). |
| `pilot/data/DATA_DICTIONARY.md` | Every input and derived file, with source and licence. |
| `pilot/data/processed/` | Cleaned building table (`buildings_labelled.csv/.gpkg`) and terrain features (`features.csv`). Lets steps 04–30 run without the raw downloads. |
| `pilot/results/` | All result tables (`*.csv`), run logs (`log_*.txt`) and figures (`*.png`). Key files below. |
| `pilot/results/audit/` | Visual label audit: sample (`label_audit.csv`, 200 rows with the final `judgement` column), summary, review record (`REVIEW.md`), contact sheets and 80-m chips. HD chips (0.2 m/px, 390 MB) are not included; `image_urls.csv` lists the source images and `29_hd_chips.py` rebuilds them. |
| `paper/` | `paper.tex`, `refs.bib`, figures, `IEEEtran.cls`/`IEEEtranN.bst`, and the compiled `paper.pdf`. |
| `docs/` | Research plan and the four weekly reports. |

Key result files in `pilot/results/`:

- `final_auc_ci.csv`, `final_paired.csv`: held-out ROC-AUC with block-bootstrap intervals (Table VIII of the paper).
- `nearchannel_ci.csv`, `nearchannel_paired.csv`: the near-channel headline test and the HAND-only / HAND-cut baselines (Table IX).
- `scores.csv`, `best_params.csv`, `optimism_gap.csv`, `knndm_gap.csv`: nine-learner comparison and inner-vs-held-out gaps (Table X).
- `ablation_preview.csv`: feature-group ablation (Table XI).
- `attenuation_bins.csv`, `attenuation_fit.csv`, `attenuation_boot.csv`: the attenuation curve and its three bootstrap schemes (Table XII).
- `calibration.csv`, `recalibration.csv`, `damaged_share_*.csv`: calibration, intercept-only recalibration, affected share (Tables XIII–XV).
- `predictions_loro.csv`: per-building held-out predictions of the four final models.
- `cleaning_log.csv`: the three cleaning rules (Table V).

## Reproducing

```bash
cd pilot
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash run_all.sh
bash run_all.sh --audit
```

The first run downloads the raw inputs (about 2.5 GB), then runs steps 01 to 30 in about an hour on a laptop. The `--audit` flag also rebuilds the audit chips (steps 09, 11 and 29; needs internet); recorded judgements are kept.

Raw downloads (`pilot/data/raw/`) and intermediate rasters (`pilot/data/interim/`) are not in the repository; `01_download.py`
fetches them from Copernicus EMS, the Copernicus DEM bucket, Overture Maps and OpenStreetMap. The paper is compiled with
`latexmk -pdf paper.tex` in `paper/` (on a TeX Live without Helvetica/Courier metrics, see the comment at the top of `paper.tex`).

## Licences

- Code (`pilot/*.py`, `run_all.sh`): MIT, see `LICENSE`.
- Derived tables and figures carry the licences of their sources: Copernicus EMS grading (free and open, attribution),
  Overture Maps buildings (ODbL 1.0; the footprint geometries in `buildings_labelled.gpkg`), Copernicus GLO-30 (Copernicus DEM
  licence), FABDEM (CC BY-NC-SA 4.0), ICIMOD/FRTC NLCMS 2022 (CC BY 4.0), Vantor Open Data imagery (CC BY-NC 4.0; audit chips).
- Manuscript text and figures: © 2026 Rustam Niraula.

## Citation

See `CITATION.cff`.
