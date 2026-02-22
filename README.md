# ARCTraj2JaxARC

Convert [ARCTraj](https://huggingface.co/datasets/SejinKimm/ARCTraj) human reasoning trajectories into [JaxARC](https://github.com/aadimator/JaxARC)-compatible format for offline reinforcement learning.

## Overview

ARCTraj contains 10,670 human reasoning trajectories across 400 ARC-AGI-1 tasks, collected via the O2ARC web platform. This pipeline converts them into JaxARC-format transitions suitable for training RL agents.

**Key conversion challenges solved:**
- **Selection–Operation merging**: ARCTraj records Selection and Operation as separate timesteps; JaxARC combines them into a single `Action(operation_id, selection_mask)`
- **PRE-action grid semantics**: ARCTraj's `grid` field is the state *before* the action applies
- **Mid-trajectory grid resizing**: Grids can change size during a trajectory (e.g., 3×3 → 9×9)
- **Undo/Redo removal**: History actions are removed while maintaining valid state transitions
- **State continuity**: O2ARC SelectCell actions modify the grid; handled by using the next operation's grid as `next_state`
- **JaxARC-compatible padding**: Background color `-1` for padding (matching JaxARC convention)

## Action Mapping

| ARCTraj Operation | JaxARC Op ID | Description |
|---|---|---|
| Paint(color=N) | 0–9 | FILL_N |
| Move(up/down/left/right) | 20–23 | MOVE directions |
| Rotate(cw/ccw) | 24–25 | ROTATE |
| Flip(h/v) | 26–27 | FLIP |
| Copy | 28 | COPY |
| Paste | 29 | PASTE |
| ResizeGrid | 33 | RESIZE |
| Submit | 34 | SUBMIT |

## Output Format

Each trajectory is a dict with:

| Field | Shape | Type | Description |
|---|---|---|---|
| `states` | (T, 30, 30) | int32 | Padded grid states (padding=-1) |
| `state_masks` | (T, 30, 30) | bool | Valid cell masks |
| `next_states` | (T, 30, 30) | int32 | Next grid states |
| `next_state_masks` | (T, 30, 30) | bool | Next state valid masks |
| `actions` | (T,) | int32 | JaxARC operation IDs (0–34) |
| `selections` | (T, 30, 30) | bool | Selection masks |
| `rewards` | (T,) | float32 | Similarity-based rewards |
| `grid_hs` | (T,) | int32 | Per-step grid height |
| `grid_ws` | (T,) | int32 | Per-step grid width |
| `task_id` | — | str | ARC task identifier |
| `success` | — | bool | Whether trajectory solved the task |
| `has_orphan_paste` | — | bool | Paste without preceding Copy (unusable for training) |

## Usage

### Reconstruct data from split files

```bash
cat output/arctraj_jaxarc.pkl.part_* > output/arctraj_jaxarc.pkl
```

### Convert from scratch

```python
python convert.py
```

Edit the paths in `convert.py` to point to your ARCTraj CSV and ARC-AGI-1 data.

### PyTorch Dataset

```python
from dataset import ARCTrajDataset, collate_trajectories
from torch.utils.data import DataLoader

dataset = ARCTrajDataset("output/arctraj_jaxarc.pkl")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_trajectories)

for batch in loader:
    states = batch["states"]         # (B, T, 30, 30)
    actions = batch["actions"]       # (B, T)
    selections = batch["selections"] # (B, T, 30, 30)
    rewards = batch["rewards"]       # (B, T)
    # ...
```

### Filtering

```python
import pickle

with open("output/arctraj_jaxarc.pkl", "rb") as f:
    data = pickle.load(f)

# Imitation learning (success + clean only)
clean_success = [t for t in data if t["success"] and not t["has_orphan_paste"]]  # 7,110

# Offline RL (include failures, exclude broken trajectories)
all_clean = [t for t in data if not t["has_orphan_paste"]]  # 9,684
```

## Conversion Stats

- **Total converted**: 10,193 / 10,670 trajectories
- **Unique tasks**: 400
- **Avg trajectory length**: 8.1 steps
- **Length range**: 1–194 steps

| | Clean | Orphan Paste | Total |
|---|---:|---:|---:|
| **Success** | 7,110 | 391 | 7,501 |
| **Failed** | 2,574 | 118 | 2,692 |
| **Total** | **9,684** | **509** | **10,193** |

## References

- [ARCTraj: A Dataset of Human Solution Trajectories for the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2511.11079)
- [JaxARC: A JaxMARL Environment for ARC](https://arxiv.org/abs/2601.17564)
- [ARC-AGI](https://arcprize.org/)
