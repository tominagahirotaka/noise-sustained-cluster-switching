# Code for "Noise-Sustained Slow Switching between Clusters in Globally Coupled Excitable Elements with Repulsive Interactions"

Author: Hirotaka Tominaga. The code is distributed under the MIT License.
The associated data archive is version 1.0.0, DOI: https://doi.org/10.5281/zenodo.24297764 (the DOI resolves after publication).

This repository contains the scripts from which the figures, tables and the numbers quoted in the paper and its
supplementary material can be regenerated from stored numerical outputs (Stage A), and the scripts that produced those
outputs (Stage B). The stored outputs are archived separately on Zenodo under the same version
number. Code: MIT License (`LICENSE`); data: CC BY 4.0.

## Getting the data

Download and extract the Zenodo archive `paper3-data-v1.0.0.zip` and copy its `analysis/` directory into the
root of this repository, so that the scripts find their inputs under `analysis/`. The archive also contains
`reference_outputs/`, the figure files and JSON outputs used in the paper, for comparison with the regenerated files.

```
pip install -r requirements.txt
cp -r <path to the Zenodo archive>/analysis .
```

Paths below are relative to the repository root.

## Two levels of reproduction

**Stage A: regeneration from stored outputs (no numerical integration).** The figures, Tables I-III and the numbers
quoted in the text are recomputed from the files under `analysis/`. This takes about five minutes. It was checked on
2026-10-31 by placing this code and the Zenodo data in a new empty directory: all figure PNGs were reproduced pixel for
pixel, and `aggregates.json` and `pathB_summary.json` were reproduced exactly apart from the time stamp, the command
line and the run time (compared with `reference_outputs/` of the data archive). Checked again on 2026-11-03, after the PDF versions of the
figures were set to 600 dpi, with the same result (also with Python 3.13.13 and NumPy 2.4.6).

A further check on 2026-11-17 reproduced the numerical aggregates (excluding run metadata) and five of seven PNGs pixel for pixel. Two PNGs (Figs. 1 and 3) differed in image dimensions and layout; the values in the regenerated aggregates agreed.

**Stage B: recomputation from starting states.** The scripts that produced the stored outputs are included, with the
starting states, random seeds, integration method and run lengths (Sec. II and Supplementary Material, Secs. S1 and
S6, and the docstrings of the scripts). Stage B has not been verified from this repository. Long runs of the noisy
full system and of the tree representation are sensitive to rounding: runs repeated on another computer gave the same
groups and classifications but different trailing digits of the stored deviations (Supplementary Material, Sec. S1).
Stage B reproductions are therefore expected to recover the classifications, counts and rates of the paper, rather
than bitwise identical records.

## Stage A: commands

```
python scripts/paper3_aggregates.py        # residences, growth rates, predictions, fits, D-cycle lengths
                                           #   -> analysis/paper3_figs/aggregates.json (about 5 min)
python scripts/paper3_pathB_summary.py     # Supplementary Sec. S3 -> analysis/switching_check/rerun_states/pathB_summary.json
python scripts/plot_paper3_figs.py         # Figs. 1-6 and S1 -> analysis/paper3_figs/fig*.{pdf,png}
```

`plot_paper3_figs.py` reads `aggregates.json`, so run `paper3_aggregates.py` first. The collective multipliers in Table I (`node_collective.json`) come from `python scripts/paper3_node_collective.py`,
which integrates one period of the variational equation from the stored node orbits (a few seconds). The figure files are named after
the working numbers of the figures: `fig1_switching_394` (Fig. 1), `fig2_network` (Fig. 2), `fig3_departure_time`
(Fig. 3), `fig4_partitions` (Fig. 4), `fig5_run88` (Fig. 5), `fig7_selection` (Fig. 6), `figS1_noiseless88` (Fig. S1).

| Item | Inputs (under `analysis/`) | Produced by (Stage B) |
|---|---|---|
| Fig. 1 | `itinerancy_n50/trace_0394.npz`, `heteroclinic_n50/ejections_0394.json` | `itinerancy_n50.py trace`, `ejections_n50.py` |
| Fig. 2 | none (schematic) | |
| Fig. 3, Sec. III C | `itinerancy_n50/trace_*.npz`, `noiseless_n50/nl_03*_t5000.npz`, `noiseless_n50/residences.json`, `heteroclinic_n50/node_*.json` | `itinerancy_n50.py`, `noiseless_n50.py`, `noiseless_switches.py`, `heteroclinic_n50.py` |
| Fig. 4, Sec. IV | `heteroclinic_n50/balanced_partitions.json`, `equal_partition.json`, `n50_grid.json`, `switching_check/renoise_summary.json` | `balanced_partitions_n.py`, `paper3_equal_partition.py`, `renoise_summary.py` |
| Table I, Sec. III B | `heteroclinic_n50/node_*.json`, `node_collective.json`, `n50_web_analysis.json`, `ejections_*.json` | `heteroclinic_n50.py`, `paper3_node_collective.py`, `n50_web_analysis.py`, `ejections_n50.py` |
| Fig. 5, Table II, Sec. V | `switching_check/rn_0088_*t80000_states_z.npy`, `return_rate_0088_O.json`, `switch_path_0088.json`, `noise_scaling_0088.json` | `switching_renoise.py --states`, `return_rate.py`, `switch_path_check.py`, `noise_scaling_check.py` |
| Sec. V D, Fig. S1 | `noiseless_n50/stageB_classify.json`, `cycles_rn_0088_*.json`, `longdouble_*.json`, `stageA.json` | `noiseless_n50.py`, `stageB_classify.py`, `noiseless_cycles.py`, `longdouble_check.py`, `stageA_compare.py` |
| Fig. 6, Table III, Sec. VI, Tables S7, S8 | `noiseless_n50/stageB_classify*.json`, `stageC_classify_*.json`, `stageB_cycle_detail_mid2.json`, `period_vs_eta.json`, `xhost_*.json` | `noiseless_n50.py --eta --seed`, `stageB_classify.py`, `stageB_cycle_detail.py`, `period_vs_eta.py`, `xhost_compare.py` |
| Sec. III D, S2, S3 | `switching_check/rerun_states/*`, `heteroclinic_n50/firstdep/*` | `rerun_periodmin.py`, `rerun_moves.py`, `pathB_firing_offsets.py`, `first_departure_n50.py`, `first_departure_analysis.py` |
| Sec. II C, S1 | `noiseless_n50/stageA.json`, `test_noise_kick.json`, `longdouble_*.json`, `xhost_*.json`, `paper3_figs/check394_double.json` | `stageA_compare.py`, `test_noise_kick.py`, `longdouble_check.py`, `xhost_compare.py`, `paper3_check394_double.py` |
| Table S9 (origin) | `switching_check/selected.json`, `sw_*.json` | `eps_cluster_ensemble.py`, `switching_check.py select`, `switching_check.py run` |

## Examples and their origin

| Run | Role | N | eps | eps K | Seed of the ensemble run |
|---|---|---|---|---|---|
| 379, 386 | Sec. III | 50 | 0.25 | -0.9 | 5025, 5040 |
| 393, 394 | Sec. III, Figs. 1, 3 | 50 | 0.25 | -1.0 | 5042, 5048 |
| 397 | Sec. III | 50 | 0.25 | -1.2 | 5025 |
| 358, 363, 364, 365, 367 | Sec. III D, S2 | 50 | 0.25 | -0.8 | 5001, 5016, 5020, 5022, 5025 |
| 396, 398 | Sec. III D, S3 | 50 | 0.25 | -1.2 | 5016, 5042 |
| 88 | Secs. V, VI | 400 | 0.2 | -0.9 | 5002 |

The run numbers are positions in the list of 418 runs selected from the ensemble (`analysis/switching_check/selected.json`,
largest N first), and the numbers in file names refer to them.

- Ensemble (the full records are not archived): N in {50, 100, 200, 400, 1000}, eps in {0.2, ..., 0.4}, eps K in
  {-0.4, ..., -1.0, -1.2}, seeds 5001-5050; initial state from NumPy `default_rng(seed)` (PCG64), x_i ~ U(-2.5, 2.5),
  then y_i ~ U(-2, 2); fourth-order Runge-Kutta, step 0.005, to t = 4e4, by `src/fhn_replay.c`. `selected.json` holds
  the seed, partition and final state of every selected run.
- Continuation (`switching_check.py run`): 5e4 time units from the final state after deviations of sd 1e-6 (PCG64 with
  seed equal to the run number), no repeated noise.
- Runs with noise (`switching_renoise.py`): from the final state of the continuation, deviations of sd 1e-6, then every
  10 time units Gaussian increments of sd eta, from `default_rng(10000 + run number)`. The noise sequences for different
  eta are the same up to the factor eta. `itinerancy_n50.py trace` repeats these runs and stores the full state every
  time unit.
- Tree runs (`noiseless_n50.py`): N = 50 from t = 5000 of the trace at eta = 1e-10; run 88 from S8 (record 1600 of
  `rn_0088_n1e-08_t80000_states_z.npy`) or S6 (record 1300 of `rn_0088_n1e-06_t80000_states_z.npy`); weak noise from
  `default_rng(seed)` with seeds 1-8.

## Software

- Regeneration (Stage A) on 2026-10-31: Python 3.12.15, NumPy 2.5.4, SciPy 1.17.1, matplotlib 3.11.2, Pillow 13.0.0,
  Numba 0.68.0 (`requirements.txt`).
- Original computations: Python 3.12.14 or 3.12.15, NumPy 2.5.4 and Numba 0.68.0 on all computers; SciPy 1.18.1 for the
  Stage B weak-noise runs on one computer and 1.17.1 otherwise (SciPy is used only to group elements into clusters).
- The Stage B weak-noise runs used an earlier version of `noiseless_n50.py` (SHA-256 beginning 487bd0af) that differed
  from the included version (ae6aceb7) in the format of the output file names; the earlier file is not kept.
- The original computations were distributed over several Linux machines; some file names carry the short name of the
  machine (`horse`, `boar`). `eps_cluster_ensemble.py` is the public version of the paper-2 repository: its default
  `--hosts` is `local:4` and its remote directory name is generic; with `--hosts local:N` it does not use ssh.
- Docstrings and comments refer to the author's working notes (for example `docs/RESULTS_NOTES.md`) and to file names
  of the original project; these notes are not part of this repository.

## Outputs that are not archived

The tree runs of run 88 without noise and with weak noise (`analysis/noiseless_n50/nl_rn_0088_*.npz`, about 43 MB
each, 2.7 GB in all) are not archived, except the 11 shorter ones used in the checks of Supplementary Sec. S1; their classifications and cycle data are (`stageB_classify*.json`, `stageC_classify_*.json`), and
the list of the omitted files is `left_out_stage_B.tsv` in the data archive. They can be recomputed in Stage B, or are
available from the author on request.
