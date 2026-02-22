"""Main conversion pipeline: ARCTraj CSV → JaxARC-format trajectory dataset.

Handles:
- PRE-action grid semantics (state = this grid, next_state = next grid)
- Dynamic grid resizing mid-trajectory
- Undo/Redo removal with correct state reconstruction
- Selection accumulation across consecutive Selection actions
"""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mapping import convert_trajectory
from utils import pad_grid, pad_selection, compute_similarity, MAX_GRID_SIZE


def load_ground_truth(arc_dir: str) -> dict[str, list[np.ndarray]]:
    """Load ARC-AGI-1 ground truth test outputs."""
    gt = {}
    for fname in os.listdir(arc_dir):
        if not fname.endswith(".json"):
            continue
        task_id = fname.replace(".json", "")
        with open(os.path.join(arc_dir, fname)) as f:
            task = json.load(f)
        gt[task_id] = [np.array(pair["output"], dtype=np.int32) for pair in task["test"]]
    return gt


def check_success(actions: list[dict], gt_outputs: list[np.ndarray]) -> bool:
    """Check if the trajectory's submitted grid matches any ground truth."""
    if not actions:
        return False
    # Find the last Submit action's next grid (= grid after Submit)
    # Since grid is PRE-action, the Submit grid is the state before submission.
    # But we already verified success using the raw grids in the CSV.
    # Here we check by finding the Submit and looking at the grid it submitted.
    last = actions[-1]
    if last.get("operation") != "Submit":
        return False

    # The submitted grid IS the Submit action's grid (the PRE-action state
    # is what the user sees and decides to submit)
    submitted = np.array(last.get("grid", [[]]), dtype=np.int32)
    return any(np.array_equal(submitted, gt) for gt in gt_outputs)


def get_target_grid(task_id: str, gt: dict) -> np.ndarray | None:
    """Get the first test output grid for a task."""
    outputs = gt.get(task_id, [])
    return outputs[0] if outputs else None


def convert_single_trajectory(
    actions: list[dict],
    target: np.ndarray,
) -> dict | None:
    """Convert a single ARCTraj trajectory to JaxARC format.

    Returns a dict with padded arrays ready for training, or None if invalid.
    Each step stores its own grid dimensions to handle mid-trajectory resizes.
    """
    transitions = convert_trajectory(actions, target_grid=target)
    if not transitions:
        return None

    n_steps = len(transitions)

    # Build padded arrays for the trajectory
    states = np.zeros((n_steps, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int32)
    state_masks = np.zeros((n_steps, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=bool)
    next_states = np.zeros((n_steps, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=np.int32)
    next_state_masks = np.zeros((n_steps, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=bool)
    action_ops = np.zeros(n_steps, dtype=np.int32)
    selections = np.zeros((n_steps, MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=bool)
    rewards = np.zeros(n_steps, dtype=np.float32)
    grid_hs = np.zeros(n_steps, dtype=np.int32)
    grid_ws = np.zeros(n_steps, dtype=np.int32)

    # Pad target for similarity computation
    target_padded, target_mask = pad_grid(target)

    for t, trans in enumerate(transitions):
        s_padded, s_mask = pad_grid(trans["state"])
        ns_padded, ns_mask = pad_grid(trans["next_state"])
        sel_padded = pad_selection(trans["selection"])

        states[t] = s_padded
        state_masks[t] = s_mask
        next_states[t] = ns_padded
        next_state_masks[t] = ns_mask
        action_ops[t] = trans["action_op"]
        selections[t] = sel_padded
        grid_hs[t] = trans["grid_h"]
        grid_ws[t] = trans["grid_w"]

        # Reward = similarity improvement toward target
        sim_before = compute_similarity(s_padded, target_padded, target_mask)
        sim_after = compute_similarity(ns_padded, target_padded, target_mask)
        rewards[t] = sim_after - sim_before

    return {
        "states": states,               # (T, 30, 30) int32
        "state_masks": state_masks,     # (T, 30, 30) bool
        "next_states": next_states,     # (T, 30, 30) int32
        "next_state_masks": next_state_masks,  # (T, 30, 30) bool
        "actions": action_ops,          # (T,) int32
        "selections": selections,       # (T, 30, 30) bool
        "rewards": rewards,             # (T,) float32
        "grid_hs": grid_hs,            # (T,) int32 — per-step grid height
        "grid_ws": grid_ws,            # (T,) int32 — per-step grid width
    }


def run_conversion(
    csv_path: str,
    arc_dir: str,
    output_dir: str,
    success_only: bool = True,
):
    """Run the full conversion pipeline."""
    print("Loading ground truth...")
    gt = load_ground_truth(arc_dir)
    print(f"  Loaded {len(gt)} ARC-AGI-1 tasks")

    print("Loading ARCTraj CSV...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} trajectories")

    os.makedirs(output_dir, exist_ok=True)

    converted = []
    stats = {
        "total": 0,
        "skipped_no_task": 0,
        "skipped_failed": 0,
        "skipped_convert_error": 0,
        "converted": 0,
        "task_counts": {},
    }

    for i in tqdm(range(len(df)), desc="Converting"):
        row = df.iloc[i]
        task_id = row["taskId"]
        stats["total"] += 1

        if not isinstance(task_id, str) or task_id not in gt:
            stats["skipped_no_task"] += 1
            continue

        try:
            actions = json.loads(row["actionSequence"])
        except (json.JSONDecodeError, TypeError):
            stats["skipped_convert_error"] += 1
            continue
        if not isinstance(actions, list):
            stats["skipped_convert_error"] += 1
            continue

        if success_only and not check_success(actions, gt[task_id]):
            stats["skipped_failed"] += 1
            continue

        target = get_target_grid(task_id, gt)
        if target is None:
            stats["skipped_no_task"] += 1
            continue

        result = convert_single_trajectory(actions, target)
        if result is None:
            stats["skipped_convert_error"] += 1
            continue

        result["task_id"] = task_id
        result["log_id"] = int(row["logId"])
        result["success"] = True if success_only else check_success(actions, gt[task_id])
        converted.append(result)

        stats["converted"] += 1
        stats["task_counts"][task_id] = stats["task_counts"].get(task_id, 0) + 1

    # Save converted data
    output_path = os.path.join(output_dir, "arctraj_jaxarc.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(converted, f)

    # Save metadata
    meta = {
        "stats": stats,
        "num_trajectories": len(converted),
        "num_tasks": len(stats["task_counts"]),
        "success_only": success_only,
        "max_grid_size": MAX_GRID_SIZE,
        "num_operations": 35,
    }
    if converted:
        lengths = [t["actions"].shape[0] for t in converted]
        meta["avg_trajectory_length"] = float(np.mean(lengths))
        meta["min_trajectory_length"] = int(np.min(lengths))
        meta["max_trajectory_length"] = int(np.max(lengths))

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Converted: {stats['converted']}")
    print(f"  Skipped (no task): {stats['skipped_no_task']}")
    print(f"  Skipped (failed): {stats['skipped_failed']}")
    print(f"  Skipped (error): {stats['skipped_convert_error']}")
    print(f"  Unique tasks: {len(stats['task_counts'])}")
    if converted:
        print(f"  Avg trajectory length: {meta['avg_trajectory_length']:.1f}")
        print(f"  Min/Max length: {meta['min_trajectory_length']}/{meta['max_trajectory_length']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    CSV_PATH = "/home/sejin/data/ARCTraj.csv"
    ARC_DIR = "/home/sejin/data/ARC-AGI-1/data/training"
    OUTPUT_DIR = "/home/sejin/ARCTraj2JaxARC/output"

    run_conversion(CSV_PATH, ARC_DIR, OUTPUT_DIR, success_only=False)
