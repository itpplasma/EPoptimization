# Repository Guidelines

## Project Structure & Module Organization

- `main.py`: primary EP + equilibrium optimization driver (MPI-aware); produces
  output folders like `out_s*_NFP*` and CSV logs `output_*.csv`.
- `initial_configs/`: VMEC input files used as starting points (e.g.
  `initial_configs/input.nfp2_QA`).
- `Alan_objectives.py`: objective/penalty utilities (mirror ratio, elongation).
- `plot_opt.py`, `vmecPlot2.py`: post-processing and plotting helpers.
- `regression_test.py`, `test_search_global_minimum.py`: standalone scripts for
  robustness checks / optimization experiments.

## Build, Test, and Development Commands

This repo is script-driven (no package build step). Use a Python environment
that has the scientific stack plus VMEC + NEAT installed.

- Install dependencies (examples):
  - `python3 -m pip install numpy scipy pandas matplotlib mpi4py simsopt booz_xform`
  - VMEC extension (required by `simsopt.mhd.Vmec`): `python3 -m pip install git+ssh://git@github.com/hiddenSymmetries/VMEC2000.git`
  - NEAT (required by `from neat...`): `python3 -m pip install -e ../NEAT`
- Run optimization: `mpirun -n 4 python3 main.py` (or `python3 main.py` for
  serial experiments).
- Plot results (after an optimization run): `python3 plot_opt.py`.

## Coding Style & Naming Conventions

- Python, 4-space indentation, keep lines reasonably short.
- Prefer `pathlib.Path` over manual string path joins in new code.
- Keep scripts runnable from the repository root (assume `initial_configs/` is
  available via relative paths).

## Testing Guidelines

There is no pytest suite. Use quick smoke checks before larger runs:

- VMEC + SIMSOPT sanity run (writes to `/tmp`): `python3 -c "from simsopt.mhd import Vmec; import os; os.chdir('/tmp'); Vmec('path/to/initial_configs/input.nfp2_QA', verbose=False).run()"`
- Import sanity: `python3 -c "from neat.fields import Simple; from simsopt.mhd import Vmec"`

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, imperative, and capitalized (e.g.
  Added ..., Create ..., Initial commit); follow that style.
- PRs should describe the physics intent, list runtime/environment assumptions
  (MPI size, key parameters), and include plots/CSV excerpts when they help
  reviewers reproduce results.
