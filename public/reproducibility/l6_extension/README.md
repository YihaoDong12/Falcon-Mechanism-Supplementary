# Periodic-L6 extension

This package supports the periodic-L6 extension reported in the updated manuscript. It is separate from the verified 54-dimensional base event at evaluation 1,074,004.

## Contents

- `model/`: analytical and optimization code with support for the periodic L6 law.
- `input/optimization_input.json`: frozen extension input contract.
- `results/best_event.json`: winning extended design event.
- `results/phasewise_replay.csv`: phase-resolved target and generated trajectories.
- `results/formal_best_history.csv`: formal incumbent history.
- `results/link_screening_comparison.csv`: equal-budget comparison of the five tested links.
- `results/independent_replay_audit.json`: independent replay checks.

The extension reached wingtip, wrist, and coupled RMSE values of 39.04, 39.60, and 39.32 mm. The optimized L6 law spanned 246.32–247.17 mm. Because all coordinates were reoptimized after introducing periodic L6, the reported gain belongs to the enlarged design space and is not an isolated causal effect of L6.
