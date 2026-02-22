"""Investigate MOVE no-op cases in the raw ARCTraj data."""

import json
import pickle
import numpy as np
import pandas as pd

# Load raw CSV to check original actions
print("Loading ARCTraj CSV...")
df = pd.read_csv("/home/sejin/IntentionLearning/dataset/ARCTraj/ARCTraj.csv")

# Load converted data
print("Loading converted data...")
with open("/home/sejin/arctraj2jaxarc/output/arctraj_jaxarc.pkl", "rb") as f:
    data = pickle.load(f)

# Find trajectories with MOVE no-ops in converted data
# MOVE_UP=20, MOVE_DOWN=21, MOVE_LEFT=22, MOVE_RIGHT=23
move_ops = {20, 21, 22, 23}
op_names = {20: "MOVE_UP", 21: "MOVE_DOWN", 22: "MOVE_LEFT", 23: "MOVE_RIGHT"}

# Collect log_ids with MOVE no-ops
noop_examples = []
changed_examples = []

for traj in data:
    T = traj["actions"].shape[0]
    for t in range(T):
        op = int(traj["actions"][t])
        if op not in move_ops:
            continue
        state = traj["states"][t]
        ns = traj["next_states"][t]
        gh = int(traj["grid_hs"][t])
        gw = int(traj["grid_ws"][t])
        is_noop = np.array_equal(state, ns)

        entry = {
            "log_id": traj["log_id"],
            "task_id": traj["task_id"],
            "step": t,
            "op": op,
            "op_name": op_names[op],
            "grid_h": gh,
            "grid_w": gw,
            "state_unpadded": state[:gh, :gw].copy(),
            "ns_unpadded": ns[:gh, :gw].copy(),
        }

        if is_noop and len(noop_examples) < 5:
            noop_examples.append(entry)
        elif not is_noop and len(changed_examples) < 5:
            changed_examples.append(entry)

        if len(noop_examples) >= 5 and len(changed_examples) >= 5:
            break

# Now look at the RAW ARCTraj actions for these examples
print("\n" + "=" * 60)
print("MOVE NO-OP EXAMPLES (state == next_state in converted data)")
print("=" * 60)

for ex in noop_examples:
    log_id = ex["log_id"]
    row = df[df["logId"] == log_id].iloc[0]
    actions = json.loads(row["actionSequence"])

    # Find the Move action in raw data
    move_actions = [a for a in actions if a.get("operation") == "Move"]

    print(f"\n--- log_id={log_id}, task={ex['task_id']}, step={ex['step']}, {ex['op_name']} ---")
    print(f"  Grid size: {ex['grid_h']}x{ex['grid_w']}")
    print(f"  State (unpadded):")
    for r in range(ex['grid_h']):
        print(f"    {ex['state_unpadded'][r].tolist()}")
    print(f"  Next_state (unpadded):")
    for r in range(ex['grid_h']):
        print(f"    {ex['ns_unpadded'][r].tolist()}")

    # Show raw Move actions
    print(f"  Raw Move actions in this trajectory ({len(move_actions)} total):")
    for i, ma in enumerate(move_actions[:3]):
        print(f"    [{i}] direction={ma.get('direction')}, object={len(ma.get('object', []))} cells")
        g = np.array(ma.get("grid", [[]]), dtype=np.int32)
        print(f"        grid shape: {g.shape}")
        # Find the NEXT action after this Move
        idx_in_actions = actions.index(ma)
        if idx_in_actions + 1 < len(actions):
            next_a = actions[idx_in_actions + 1]
            ng = np.array(next_a.get("grid", [[]]), dtype=np.int32)
            print(f"        next action: op={next_a.get('operation')}, grid shape: {ng.shape}")
            if g.shape == ng.shape:
                diff = (g != ng)
                if diff.any():
                    print(f"        grid CHANGED: {diff.sum()} cells differ")
                    # Show which cells changed
                    ys, xs = np.where(diff)
                    for j in range(min(5, len(ys))):
                        print(f"          ({ys[j]},{xs[j]}): {g[ys[j],xs[j]]} -> {ng[ys[j],xs[j]]}")
                else:
                    print(f"        grid UNCHANGED between Move and next action")

print("\n\n" + "=" * 60)
print("MOVE CHANGED EXAMPLES (state != next_state)")
print("=" * 60)

for ex in changed_examples:
    log_id = ex["log_id"]
    row = df[df["logId"] == log_id].iloc[0]
    actions = json.loads(row["actionSequence"])

    print(f"\n--- log_id={log_id}, task={ex['task_id']}, step={ex['step']}, {ex['op_name']} ---")
    print(f"  Grid size: {ex['grid_h']}x{ex['grid_w']}")
    print(f"  State (unpadded):")
    for r in range(ex['grid_h']):
        print(f"    {ex['state_unpadded'][r].tolist()}")
    print(f"  Next_state (unpadded):")
    for r in range(ex['grid_h']):
        print(f"    {ex['ns_unpadded'][r].tolist()}")

    # Show diff
    diff = ex['state_unpadded'] != ex['ns_unpadded']
    ys, xs = np.where(diff)
    print(f"  Changed cells: {len(ys)}")
    for j in range(min(10, len(ys))):
        print(f"    ({ys[j]},{xs[j]}): {ex['state_unpadded'][ys[j],xs[j]]} -> {ex['ns_unpadded'][ys[j],xs[j]]}")


# Now let's do a deeper analysis: for MOVE no-ops, check what actually
# happened in the RAW data between this Move and the next operation
print("\n\n" + "=" * 60)
print("DEEPER ANALYSIS: Raw grid changes around MOVE actions")
print("=" * 60)

# Sample a few trajectories with Move operations
sample_count = 0
move_raw_changed = 0
move_raw_unchanged = 0

for _, row in df.iterrows():
    if sample_count > 200:
        break
    actions = json.loads(row["actionSequence"])

    for i, a in enumerate(actions):
        if a.get("operation") != "Move":
            continue

        # Check: does the grid IMMEDIATELY after this Move differ from Move's grid?
        g = np.array(a.get("grid", [[]]), dtype=np.int32)

        # Find next action's grid
        if i + 1 < len(actions):
            ng = np.array(actions[i+1].get("grid", [[]]), dtype=np.int32)
            if g.shape == ng.shape:
                if np.array_equal(g, ng):
                    move_raw_unchanged += 1
                else:
                    move_raw_changed += 1
                    if move_raw_changed <= 3:
                        print(f"\n  Raw Move that changed grid (logId={row['logId']}):")
                        print(f"    direction={a.get('direction')}")
                        diff = g != ng
                        print(f"    {diff.sum()} cells changed")
                        ys, xs = np.where(diff)
                        for j in range(min(5, len(ys))):
                            print(f"      ({ys[j]},{xs[j]}): {g[ys[j],xs[j]]} -> {ng[ys[j],xs[j]]}")
            else:
                # Grid size changed
                pass
    sample_count += 1

total_raw_move = move_raw_changed + move_raw_unchanged
print(f"\n  Raw Move analysis (first 200 trajectories):")
print(f"    Grid changed after Move: {move_raw_changed}/{total_raw_move} ({move_raw_changed/total_raw_move*100:.1f}%)")
print(f"    Grid unchanged after Move: {move_raw_unchanged}/{total_raw_move} ({move_raw_unchanged/total_raw_move*100:.1f}%)")
