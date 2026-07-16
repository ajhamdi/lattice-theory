# Lattice Theory — Data & Code

Data and code accompanying the research paper (under review). This repository
holds everything needed to reproduce every fitted number, registered forecast,
and figure in the paper. It contains **only** the datasets, the fitting/plotting
code, and the generated figures — not the manuscript itself.

Each script states the quantity it fits and the null model it tests in its
own docstring; the paper is where the results are interpreted.

## Layout

```
code/       Python scripts: data fits, forward forecasts, and figure generation
data/       Input datasets (CSV, with source provenance in each file header)
            and the fitted-parameter outputs (fit_*.json)
figures/    Generated figures (PNG), regenerable from code + data
```

### `code/`

| Script | What it does |
| --- | --- |
| `fit_avrami_democratization.py` | Case A — Avrami/JMAK transformation kinetics vs logistic/linear nulls |
| `fit_kenostate_arrhenius.py`    | Case B — Arrhenius vacancy law for conflict counts vs systemic temperature |
| `fit_weibull_empires.py`        | Case C — Weibull weakest-link statistics of empire lifetimes |
| `fit_fatigue_paris.py`          | Case D — Paris-law fatigue crack growth / crisis-interval compression |
| `predict_2031.py`               | Near-cohort forecasts (frozen coefficients), 2026→2031 |
| `predict_horizon2046.py`        | Far-cohort forecasts (frozen coefficients), →2046 |
| `politviz.py`                   | Shared matplotlib style (imported by the others) |
| `concept_figures.py`            | Schematic concept explainer figures |
| `ms_figures.py`                 | Materials-science equation figures |
| `map_figures.py`, `geomap.py`   | Case-study maps drawn on Natural Earth borders |

Each fit script reads from `../data`, writes its fitted parameters to
`../data/fit_*.json`, and writes its plot to `../figures` (override the figure
directory with the `LT_FIGDIR` environment variable).

## Requirements

Python 3.13 with:

```
numpy
scipy
pandas
matplotlib
```

Install with `pip install -r requirements.txt`. The concept and materials-science
figure scripts need only `numpy` + `matplotlib`. `geomap.py` fetches a cached
Natural Earth GeoJSON on first run (no GIS stack required).

## Reproduce

From the repository root:

```bash
pip install -r requirements.txt

# Fits (each writes data/fit_*.json and a figure)
python code/fit_avrami_democratization.py
python code/fit_kenostate_arrhenius.py
python code/fit_weibull_empires.py
python code/fit_fatigue_paris.py

# Forecasts (read the frozen fit_*.json above)
python code/predict_2031.py
python code/predict_horizon2046.py

# Explanatory / concept / map figures
python code/concept_figures.py
python code/ms_figures.py
python code/map_figures.py
```

## Data sources

All inputs are public. Full retrieval URLs and CSV endpoints are recorded in the
`#` header of each file in `data/`; all series were pulled 2026-07-08.

| File | Source |
| --- | --- |
| `row_regimes.csv` | V-Dem "Regimes of the World" via Our World in Data |
| `number-of-armed-conflicts.csv` | UCDP (Uppsala) via Our World in Data |
| `share-of-individuals-using-the-internet.csv` | ITU / World Bank via Our World in Data |
| `population.csv` | Our World in Data population series |
| `empires.csv` | Taagepera empire-size compilations, tabulated via Wikipedia |
| `crisis_sequences.csv` | Standard diplomatic-history chronologies (July Crisis, India–Pakistan, Taiwan Strait, Nagorno-Karabakh, Kosovo–Serbia) |
| `polities.csv` | Founding years of current constitutional orders (standard reference chronology) |
| `un_admissions.csv` | UN membership records |
| `war_durations.csv` | Correlates of War interstate-war list + per-war chronologies |
| `nuclear_states.csv` | Standard nuclear-test chronology |

The `fit_*.json` files are outputs written by the scripts above.

## License

The code in `code/` is released under the MIT License (see `LICENSE`). The
datasets in `data/` are redistributed from the public sources listed above and
remain governed by their original terms.
