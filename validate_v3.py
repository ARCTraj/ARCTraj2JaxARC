"""Comprehensive validation of V3 converted data."""

import json
import pickle
import numpy as np
from collections import Counter

# Load converted data
print("Loading converted data...")
with open("/home/sejin/arctraj2jaxarc/output/arctraj_jaxarc.pkl", "rb") as f:
    data = pickle.load(f)
print(f"  Loaded {len(data)} trajectories\n")

# Load ground truth
print("Loading ground truth...")
import os
gt = {}
arc_dir = "/mnt/c/Users/DSLAB/data/ARC-AGI-1/data/training"
for fname in os.listdir(arc_dir):
    if not fname.endswith(".json"):
        continue
    task_id = fname.replace(".json", "")
    with open(os.path.join(arc_dir, fname)) as f:
        task = json.load(f)
    gt[task_id] = [np.array(pair["output"], dtype=np.int32) for pair in task["test"]]
print(f"  Loaded {len(gt)} tasks\n")

# ============================================================
# Check 1: Success verification
# ============================================================
print("=" * 60)
print("CHECK 1: Success Verification")
print("=" * 60)
check1_pass = 0
check1_fail = 0
check1_fail_details = []

for traj in data:
    task_id = traj["task_id"]
    if task_id not in gt:
        check1_fail += 1
        continue

    # Last step's next_state should match ground truth
    last_ns = traj["next_states"][-1]  # (30, 30) padded
    last_nh = int(traj["grid_hs"][-1])  # Note: grid_hs stores state dims
    last_nw = int(traj["grid_ws"][-1])

    # For the last step, next_state = target_grid (padded)
    # We need to compare the unpadded next_state with GT
    # But grid_hs/grid_ws store the state dimensions, not next_state dimensions
    # Let's use the next_state_masks to find the actual size
    ns_mask = traj["next_state_masks"][-1]
    # Find the actual height and width from mask
    rows_valid = ns_mask.any(axis=1)
    cols_valid = ns_mask.any(axis=0)
    if rows_valid.any() and cols_valid.any():
        actual_h = int(np.where(rows_valid)[0][-1]) + 1
        actual_w = int(np.where(cols_valid)[0][-1]) + 1
    else:
        actual_h, actual_w = 0, 0

    unpadded_ns = last_ns[:actual_h, :actual_w]

    matched = any(np.array_equal(unpadded_ns, g) for g in gt[task_id])
    if matched:
        check1_pass += 1
    else:
        check1_fail += 1
        if len(check1_fail_details) < 5:
            check1_fail_details.append({
                "task_id": task_id,
                "ns_shape": (actual_h, actual_w),
                "gt_shapes": [g.shape for g in gt[task_id]],
            })

print(f"  PASS: {check1_pass}/{len(data)}")
print(f"  FAIL: {check1_fail}/{len(data)}")
if check1_fail_details:
    print(f"  Sample failures:")
    for d in check1_fail_details:
        print(f"    task={d['task_id']}, ns_shape={d['ns_shape']}, gt_shapes={d['gt_shapes']}")

# ============================================================
# Check 2: State Continuity (next_state[t] == state[t+1])
# ============================================================
print("\n" + "=" * 60)
print("CHECK 2: State Continuity (next_state[t] == state[t+1])")
print("=" * 60)
check2_pass = 0
check2_fail = 0
check2_fail_trajs = 0
check2_total_transitions = 0

for traj in data:
    T = traj["actions"].shape[0]
    traj_ok = True
    for t in range(T - 1):
        check2_total_transitions += 1
        ns = traj["next_states"][t]
        s_next = traj["states"][t + 1]
        if np.array_equal(ns, s_next):
            check2_pass += 1
        else:
            check2_fail += 1
            traj_ok = False
    if not traj_ok:
        check2_fail_trajs += 1

print(f"  Transitions checked: {check2_total_transitions}")
print(f"  PASS: {check2_pass}/{check2_total_transitions}")
print(f"  FAIL: {check2_fail}/{check2_total_transitions}")
print(f"  Trajectories with failures: {check2_fail_trajs}/{len(data)}")

# Also check mask continuity
mask_pass = 0
mask_fail = 0
for traj in data:
    T = traj["actions"].shape[0]
    for t in range(T - 1):
        ns_mask = traj["next_state_masks"][t]
        s_next_mask = traj["state_masks"][t + 1]
        if np.array_equal(ns_mask, s_next_mask):
            mask_pass += 1
        else:
            mask_fail += 1

print(f"  Mask continuity PASS: {mask_pass}, FAIL: {mask_fail}")

# ============================================================
# Check 3: No-op Analysis (state[t] == next_state[t])
# ============================================================
print("\n" + "=" * 60)
print("CHECK 3: No-op Analysis (state == next_state)")
print("=" * 60)

op_names = {
    0: "FILL_0", 1: "FILL_1", 2: "FILL_2", 3: "FILL_3", 4: "FILL_4",
    5: "FILL_5", 6: "FILL_6", 7: "FILL_7", 8: "FILL_8", 9: "FILL_9",
    20: "MOVE_UP", 21: "MOVE_DOWN", 22: "MOVE_LEFT", 23: "MOVE_RIGHT",
    24: "ROTATE_C", 25: "ROTATE_CC", 26: "FLIP_H", 27: "FLIP_V",
    28: "COPY", 29: "PASTE", 33: "RESIZE", 34: "SUBMIT",
}

noop_by_op = Counter()
total_by_op = Counter()

for traj in data:
    T = traj["actions"].shape[0]
    for t in range(T):
        op = int(traj["actions"][t])
        total_by_op[op] += 1
        state = traj["states"][t]
        ns = traj["next_states"][t]
        if np.array_equal(state, ns):
            noop_by_op[op] += 1

print(f"  {'Operation':<15} {'Total':>7} {'No-op':>7} {'Rate':>8}")
print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*8}")
for op in sorted(total_by_op.keys()):
    name = op_names.get(op, f"OP_{op}")
    total = total_by_op[op]
    noop = noop_by_op[op]
    rate = noop / total * 100 if total > 0 else 0
    print(f"  {name:<15} {total:>7} {noop:>7} {rate:>7.1f}%")

total_all = sum(total_by_op.values())
noop_all = sum(noop_by_op.values())
print(f"  {'TOTAL':<15} {total_all:>7} {noop_all:>7} {noop_all/total_all*100:>7.1f}%")

# ============================================================
# Check 4: Selection Mask Validity
# ============================================================
print("\n" + "=" * 60)
print("CHECK 4: Selection Mask Validity")
print("=" * 60)

empty_sel = 0
total_sel = 0
full_grid_sel = 0
sel_within_mask = 0
sel_outside_mask = 0

for traj in data:
    T = traj["actions"].shape[0]
    for t in range(T):
        total_sel += 1
        sel = traj["selections"][t]
        mask = traj["state_masks"][t]

        if not sel.any():
            empty_sel += 1
        if sel.all():
            full_grid_sel += 1

        # Check selection is within valid grid area
        if (sel & ~mask).any():
            sel_outside_mask += 1
        else:
            sel_within_mask += 1

print(f"  Total selection masks: {total_sel}")
print(f"  Empty selections: {empty_sel} ({empty_sel/total_sel*100:.1f}%)")
print(f"  Full-grid selections: {full_grid_sel} ({full_grid_sel/total_sel*100:.1f}%)")
print(f"  Within valid mask: {sel_within_mask} ({sel_within_mask/total_sel*100:.1f}%)")
print(f"  Outside valid mask: {sel_outside_mask} ({sel_outside_mask/total_sel*100:.1f}%)")

# ============================================================
# Check 5: Grid Value Ranges
# ============================================================
print("\n" + "=" * 60)
print("CHECK 5: Grid Value Ranges")
print("=" * 60)

min_val = float('inf')
max_val = float('-inf')
for traj in data:
    s_min = traj["states"].min()
    s_max = traj["states"].max()
    ns_min = traj["next_states"].min()
    ns_max = traj["next_states"].max()
    min_val = min(min_val, s_min, ns_min)
    max_val = max(max_val, s_max, ns_max)

print(f"  Value range: [{min_val}, {max_val}]")
print(f"  Expected: [0, 9]")
print(f"  PASS" if 0 <= min_val and max_val <= 9 else f"  FAIL")

# ============================================================
# Check 6: Action Distribution
# ============================================================
print("\n" + "=" * 60)
print("CHECK 6: Action Distribution")
print("=" * 60)

action_counts = Counter()
for traj in data:
    for a in traj["actions"]:
        action_counts[int(a)] += 1

total_actions = sum(action_counts.values())
print(f"  Total actions: {total_actions}")
for op in sorted(action_counts.keys()):
    name = op_names.get(op, f"OP_{op}")
    cnt = action_counts[op]
    print(f"    {name:<15}: {cnt:>6} ({cnt/total_actions*100:>5.1f}%)")

# ============================================================
# Check 7: Trajectory lengths
# ============================================================
print("\n" + "=" * 60)
print("CHECK 7: Trajectory Length Distribution")
print("=" * 60)

lengths = [traj["actions"].shape[0] for traj in data]
print(f"  Count: {len(lengths)}")
print(f"  Mean:  {np.mean(lengths):.1f}")
print(f"  Std:   {np.std(lengths):.1f}")
print(f"  Min:   {np.min(lengths)}")
print(f"  Max:   {np.max(lengths)}")
print(f"  Median: {np.median(lengths):.0f}")

# Length histogram
bins = [0, 5, 10, 20, 50, 100, 200]
hist, _ = np.histogram(lengths, bins=bins)
for i in range(len(bins)-1):
    print(f"    [{bins[i]:>3}-{bins[i+1]:>3}): {hist[i]:>5} ({hist[i]/len(lengths)*100:>5.1f}%)")

# ============================================================
# Check 8: Last action should be SUBMIT (34)
# ============================================================
print("\n" + "=" * 60)
print("CHECK 8: Last Action is SUBMIT")
print("=" * 60)

last_submit = sum(1 for traj in data if int(traj["actions"][-1]) == 34)
print(f"  Trajectories ending with SUBMIT: {last_submit}/{len(data)}")
print(f"  PASS" if last_submit == len(data) else f"  FAIL - {len(data) - last_submit} don't end with SUBMIT")

# ============================================================
# Check 9: Reward Analysis
# ============================================================
print("\n" + "=" * 60)
print("CHECK 9: Reward Analysis")
print("=" * 60)

all_rewards = np.concatenate([traj["rewards"] for traj in data])
print(f"  Total reward values: {len(all_rewards)}")
print(f"  Mean:  {all_rewards.mean():.4f}")
print(f"  Std:   {all_rewards.std():.4f}")
print(f"  Min:   {all_rewards.min():.4f}")
print(f"  Max:   {all_rewards.max():.4f}")
print(f"  Positive: {(all_rewards > 0).sum()} ({(all_rewards > 0).mean()*100:.1f}%)")
print(f"  Zero:     {(all_rewards == 0).sum()} ({(all_rewards == 0).mean()*100:.1f}%)")
print(f"  Negative: {(all_rewards < 0).sum()} ({(all_rewards < 0).mean()*100:.1f}%)")

# Check cumulative reward for successful trajectories
cum_positive = 0
cum_total = 0
for traj in data:
    if traj["success"]:
        cum = traj["rewards"].sum()
        cum_total += 1
        if cum > 0:
            cum_positive += 1

print(f"  Successful trajectories with positive cumulative reward: {cum_positive}/{cum_total}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
