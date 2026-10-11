# symmetry-pinn

Research code for **"Symmetry-informed PINNs when the symmetries are not known in advance"**
(learning Lie point symmetries from sparse data of a partly unknown PDE, and using them in a hybrid PINN).

Plan: see `Research_Plan_v3.pdf` and `Weekly_Timetable_v2.pdf` (in the Claude project).

## Structure

| Folder | What goes there | Plan step | Week |
|---|---|---|---|
| `scripts/` | Stand-alone scripts from the notes (`part1_symbolic.py`, `part2_data.py`) | - | 1 |
| `data/` | PDE solvers, exact solutions, sampling of sparse noisy observations | A1 | 3 |
| `discovery/` | Jets, prolongation, criterion matrix, null space, your discovery method, validation checks | A3-A5 | 3-4 |
| `pinn/` | Hybrid PINN, symmetry losses, baselines (gPINN, extra collocation) | A2, A6, A8 | 5-6 |
| `experiments/configs/` | One config file per experiment (frozen in week 7) | A9 | 7 |
| `results/` | Outputs of runs (not committed to git, see `.gitignore`) | - | 8 |
| `paper/` | Preprint and paper sources, figure scripts | - | 10 |
| `notes/` | Lab notebook (one dated entry per working day) | - | every day |
| `tests/` | Small tests that check the maths (e.g. prolongation of the scaling generator) | - | from week 3 |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/part1_symbolic.py # symbolic checks (prolongation, Burgers symmetries)
python scripts/part2_data.py     # jet-space data experiment and figures
```

## Rules we follow

1. Every figure is produced by a script, never by hand.
2. Fixed random seeds; every experiment is described by a config file.
3. Commit at least once a day with a clear message ("add Cole-Hopf solver", not "update").
4. Write in `notes/lab_notebook.md` what you ran, what you saw, what is next.
