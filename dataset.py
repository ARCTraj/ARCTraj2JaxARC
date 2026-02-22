"""PyTorch Dataset for converted ARCTraj → JaxARC trajectories."""

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ARCTrajDataset(Dataset):
    """PyTorch Dataset for JaxARC-format ARC trajectories.

    Each sample is a trajectory consisting of:
        - states:           (T, 30, 30) int32   — grid states (padded)
        - state_masks:      (T, 30, 30) bool    — valid cell masks
        - next_states:      (T, 30, 30) int32   — next grid states
        - next_state_masks: (T, 30, 30) bool    — next state valid cell masks
        - actions:          (T,) int32           — JaxARC operation IDs (0-34)
        - selections:       (T, 30, 30) bool    — selection masks
        - rewards:          (T,) float32         — similarity-based rewards
        - grid_hs:          (T,) int32           — per-step grid height
        - grid_ws:          (T,) int32           — per-step grid width
        - task_id:          str                  — ARC task identifier
        - success:          bool                 — whether trajectory solved the task
    """

    def __init__(
        self,
        data_path: str = "output/arctraj_jaxarc.pkl",
        max_seq_len: int | None = None,
        task_ids: list[str] | None = None,
    ):
        """
        Args:
            data_path: Path to the pickle file from convert.py.
            max_seq_len: If set, truncate trajectories to this length.
            task_ids: If set, filter to only these task IDs.
        """
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        if task_ids is not None:
            task_set = set(task_ids)
            self.data = [d for d in self.data if d["task_id"] in task_set]

        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | bool | int]:
        item = self.data[idx]
        T = item["actions"].shape[0]

        # Optionally truncate
        end = min(T, self.max_seq_len) if self.max_seq_len else T

        return {
            "states": torch.from_numpy(item["states"][:end]),                    # (T, 30, 30)
            "state_masks": torch.from_numpy(item["state_masks"][:end]),          # (T, 30, 30)
            "next_states": torch.from_numpy(item["next_states"][:end]),          # (T, 30, 30)
            "next_state_masks": torch.from_numpy(item["next_state_masks"][:end]),# (T, 30, 30)
            "actions": torch.from_numpy(item["actions"][:end]),                  # (T,)
            "selections": torch.from_numpy(item["selections"][:end]),            # (T, 30, 30)
            "rewards": torch.from_numpy(item["rewards"][:end]),                  # (T,)
            "grid_hs": torch.from_numpy(item["grid_hs"][:end]),                  # (T,)
            "grid_ws": torch.from_numpy(item["grid_ws"][:end]),                  # (T,)
            "task_id": item["task_id"],
            "success": item["success"],
            "seq_len": end,
        }

    def get_task_ids(self) -> list[str]:
        """Get unique task IDs in the dataset."""
        return sorted(set(d["task_id"] for d in self.data))

    def filter_by_tasks(self, task_ids: list[str]) -> "ARCTrajDataset":
        """Return a new dataset filtered to specific task IDs."""
        new_ds = ARCTrajDataset.__new__(ARCTrajDataset)
        task_set = set(task_ids)
        new_ds.data = [d for d in self.data if d["task_id"] in task_set]
        new_ds.max_seq_len = self.max_seq_len
        return new_ds

    def summary(self) -> dict:
        """Print dataset summary statistics."""
        lengths = [d["actions"].shape[0] for d in self.data]
        tasks = self.get_task_ids()
        grid_sizes = [(int(d["grid_hs"][0]), int(d["grid_ws"][0])) for d in self.data]

        info = {
            "num_trajectories": len(self.data),
            "num_tasks": len(tasks),
            "avg_length": float(np.mean(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "total_transitions": int(np.sum(lengths)),
            "unique_grid_sizes": len(set(grid_sizes)),
        }

        print(f"ARCTrajDataset Summary:")
        print(f"  Trajectories: {info['num_trajectories']}")
        print(f"  Unique tasks: {info['num_tasks']}")
        print(f"  Avg length:   {info['avg_length']:.1f} steps")
        print(f"  Range:        {info['min_length']}-{info['max_length']} steps")
        print(f"  Total transitions: {info['total_transitions']}")
        print(f"  Unique grid sizes: {info['unique_grid_sizes']}")
        return info


def collate_trajectories(batch: list[dict]) -> dict:
    """Custom collate function that pads trajectories to the same length.

    Usage:
        DataLoader(dataset, collate_fn=collate_trajectories, batch_size=32)
    """
    max_len = max(item["seq_len"] for item in batch)
    batch_size = len(batch)

    # Initialize padded tensors
    states = torch.zeros(batch_size, max_len, 30, 30, dtype=torch.long)
    state_masks = torch.zeros(batch_size, max_len, 30, 30, dtype=torch.bool)
    next_states = torch.zeros(batch_size, max_len, 30, 30, dtype=torch.long)
    next_state_masks = torch.zeros(batch_size, max_len, 30, 30, dtype=torch.bool)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    selections = torch.zeros(batch_size, max_len, 30, 30, dtype=torch.bool)
    rewards = torch.zeros(batch_size, max_len, dtype=torch.float)
    seq_lens = torch.zeros(batch_size, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    grid_hs = torch.zeros(batch_size, max_len, dtype=torch.long)
    grid_ws = torch.zeros(batch_size, max_len, dtype=torch.long)

    task_ids = []

    for i, item in enumerate(batch):
        T = item["seq_len"]
        states[i, :T] = item["states"]
        state_masks[i, :T] = item["state_masks"]
        next_states[i, :T] = item["next_states"]
        next_state_masks[i, :T] = item["next_state_masks"]
        actions[i, :T] = item["actions"]
        selections[i, :T] = item["selections"]
        rewards[i, :T] = item["rewards"]
        seq_lens[i] = T
        padding_mask[i, :T] = True
        grid_hs[i, :T] = item["grid_hs"]
        grid_ws[i, :T] = item["grid_ws"]
        task_ids.append(item["task_id"])

    return {
        "states": states,                    # (B, T_max, 30, 30)
        "state_masks": state_masks,          # (B, T_max, 30, 30)
        "next_states": next_states,          # (B, T_max, 30, 30)
        "next_state_masks": next_state_masks,# (B, T_max, 30, 30)
        "actions": actions,                  # (B, T_max)
        "selections": selections,            # (B, T_max, 30, 30)
        "rewards": rewards,                  # (B, T_max)
        "seq_lens": seq_lens,                # (B,)
        "padding_mask": padding_mask,        # (B, T_max)
        "grid_hs": grid_hs,                  # (B, T_max)
        "grid_ws": grid_ws,                  # (B, T_max)
        "task_ids": task_ids,                # list[str]
    }
