# Protein binder design

Design a binder to a target protein at specified hotspot residues. Back to the
[skill](../../SKILL.md).

## What the task is

Default to target backbone-only internal-coordinate conditioning (`C_CRDGRP`),
full target sequence (`C_SEQ`), and requested hotspots (`C_HOT`). This reduces
target-like outputs while leaving sidechains and global target pose free.
Use one group for a rigid multichain target; separate groups allow relative motion.
Preserve explicit user choices rather than converting existing inputs silently.

## Example inputs

| File | Notes |
| --- | --- |
| `docs/how-to/input_prep/examples/condition_spec_ppi.yaml` | Recommended PD-L1 internal-backbone example; sequence and hotspots retained |
| `data/benchmarks/ppi/ppi_bb_only/`, `ppi_all_atom/` | Absolute-coordinate benchmark variants; not the internal-coordinate default |

## Run

```bash
rfd4-generate \
    inputs=docs/how-to/input_prep/examples/condition_spec_ppi.yaml \
    recycle_strategy=medium/front_loaded \
    num_replicates=8 batch_size=8 out_dir=./runs/ppi \
    metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

## Task-specific overrides & metrics

- **`ppi_refolding`** — AF3 refolding + target templating + interface metrics.
- **`hotspot_conditioning`** — hotspot obedience (requires hotspot input).
- Only refolding requires the configured oracle environment (see **rfd4-proteina-evaluate**).
- `save_chain_of_interest=true` if you plan to cluster designed binders by
  structure (foldseek).

## Gotchas

- Hotspots are annotated on the target atoms in the input CIF. To build a new
  target/hotspot input, use **rfd4-proteina-annotated-cif**.
- The designed (binder) chain is the "chain of interest" for clustering; it is
  identified automatically from the conditioning annotations.
