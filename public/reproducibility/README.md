# Reproducibility archive

This archive supports the strict-phase event selected at evaluation 1,074,004 from the 12-hour cold-start run.

## Contents

- `model/fourbar3d_python.py`: analytical three-dimensional closed-loop mechanism.
- `model/fourbar_optimization.py`: optimization problem, bounds, feasibility checks, and paired objective.
- `data/optimization_input.json`: frozen 54-independent-variable input contract.
- `data/target_source_workbook.xlsx`: source workbook and initialization record.
- `data/target_initialized_equal_arc_76_mm.csv`: initialized 76-state biological target.
- `data/final_variables_and_bounds.csv`: selected event variables and bounds.
- `data/inverse_rotated_strict_trajectories.csv`: target and generated wrist/wingtip trajectories.
- `data/cma_generations.csv`: CMA-ES generation history.
- `data/partition_splits.csv`: sensitivity-partition decisions.
- `data/slsqp_calls.csv`: accepted SQP call history (legacy filename retained from the run record).
- `data/sha256_manifest.json`: hashes for the frozen optimization inputs.

Independent event replay reproduced the reported metrics to a maximum absolute difference of `6.82e-13`. The launcher did not complete checkpoint-level finalization; the archive therefore supports the selected event and does not claim a finalized terminal checkpoint or a universal global optimum.
