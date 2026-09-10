# Source layout decision

## Decision: preserve `UNIV-main/`

The original snapshot was already isolated in the top-level `UNIV-main/` directory,
not scattered among repository-root Python modules. It has therefore **not** been
moved or renamed. Its launchers assume their working directory is the source root:
`run.sh` invokes `pretrain_mcmae.py` and `configs/mcmae.yaml` by relative path, while
Python files import top-level `datasets`, `models`, `loss`, and `utils` packages.
The segmentation launchers likewise use paths relative to
`UNIV-main/SEG/MCMAE_SEG/`. Renaming the directory would also invalidate published
commands and prior documentation for no functional benefit.

New work lives in `psmaf_univ/` and task-specific top-level directories. It wraps the
original model at a module boundary and does not edit original UNIV files. To run an
original command, change into `UNIV-main/` first. To run new tools, use the repository
root (for example, `python tools/check_univ_checkpoint.py CHECKPOINT`).
