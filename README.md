# EPoptimization

This branch runs a direct collisionless alpha-loss study. It minimizes the
measured 100 ms loss fraction for 256 deterministic 3.5 MeV alpha particles
born at `s = 0.25`. Mirror ratio and maximum elongation are broad feasibility
constraints; proxy, classifier, Gamma-c, and effective-ripple quantities do
not enter the objective.

`main.py` primes constrained TuRBO/SCBO from completed direct observations.
`generate_scbo_candidates.py` maps requests through the bounded ALPES-centered
eight-dimensional chart, and `run_direct_scbo_candidate.sh` evaluates VMEC,
geometry feasibility, and the pinned SIMPLE direct metric.

The campaign contract, exact executable hashes, provenance, and cluster
orchestration live in the companion `alpha-loss-optimization-data` repository
under `runs/alpha_direct_turbo_v1/`.
