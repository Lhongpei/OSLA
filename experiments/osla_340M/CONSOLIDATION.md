# OSLA Experiment Code Consolidation

This directory collects the May 2026 OSDN experiment code that had been
split across the GPU worker checkouts.

Source mapping:

- `gpu-5`: DeltaNet/OSDN 340M configs, OSDN/APF launch scripts, evaluation
  scripts, and the DeltaNet OSGM layer updates.
- `gpu-6`: OSKDA model registration, OSKDA kernels, KDA-family configs,
  launch scripts, and benchmark helpers.
- `gpu-7`: OS-GDN post-gate-regret implementation, fused recurrent decode
  path, decode smoke tests, and OSGDN evaluation scripts.
- `phoenix2-datagen`: 1.3B experiment configs and datagen-specific launch
  scripts.

Runtime outputs such as checkpoints, model upload bundles, pid files, logs,
and generated RULER prediction JSONL files are intentionally left out of the
branch. Small JSON evaluation summaries remain trackable when they are useful
for paper tables.
