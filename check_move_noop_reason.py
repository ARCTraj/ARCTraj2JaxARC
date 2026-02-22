"""Verify hypothesis: Move no-ops happen because selected cells and destinations are both 0."""

import json
import numpy as np
import pandas as pd

print("Loading ARCTraj CSV...")
df = pd.read_csv("/home/sejin/IntentionLearning/dataset/ARCTraj/ARCTraj.csv")

DIRECTION_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# Categorize interior no-ops
all_zero_src = 0          # All selected cells are 0
all_zero_src_and_dst = 0  # All selected cells AND destinations are 0
some_nonzero = 0          # Some selected cells have non-zero values
total_interior_noop = 0

detail_examples = []

for row_idx, row in df.iterrows():
    if row_idx > 2000:
        break
    actions = json.loads(row["actionSequence"])
    for i, a in enumerate(actions):
        if a.get("operation") != "Move":
            continue
        direction = a.get("direction", "")
        objects = a.get("object", [])
        if len(objects) == 0:
            continue

        g = np.array(a.get("grid", [[]]), dtype=np.int32)
        gh, gw = g.shape
        if i + 1 >= len(actions):
            continue
        ng = np.array(actions[i + 1].get("grid", [[]]), dtype=np.int32)
        if g.shape != ng.shape or not np.array_equal(g, ng):
            continue

        # This is a no-op with selection
        dy, dx = DIRECTION_DELTA.get(direction, (0, 0))
        obj_coords = []
        for o in objects:
            x, y = o.get("x"), o.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                obj_coords.append((int(y), int(x)))

        if not obj_coords:
            continue

        # Check if at edge
        obj_ys = [c[0] for c in obj_coords]
        obj_xs = [c[1] for c in obj_coords]
        at_edge = False
        if direction == "up" and min(obj_ys) == 0:
            at_edge = True
        elif direction == "down" and max(obj_ys) == gh - 1:
            at_edge = True
        elif direction == "left" and min(obj_xs) == 0:
            at_edge = True
        elif direction == "right" and max(obj_xs) == gw - 1:
            at_edge = True
        if at_edge:
            continue

        total_interior_noop += 1

        # Check cell values
        src_values = []
        dst_values = []
        for (y, x) in obj_coords:
            if 0 <= y < gh and 0 <= x < gw:
                src_values.append(int(g[y, x]))
                ny, nx = y + dy, x + dx
                if 0 <= ny < gh and 0 <= nx < gw:
                    dst_values.append(int(g[ny, nx]))

        src_all_zero = all(v == 0 for v in src_values) if src_values else False
        dst_all_zero = all(v == 0 for v in dst_values) if dst_values else False

        if src_all_zero:
            all_zero_src += 1
            if dst_all_zero:
                all_zero_src_and_dst += 1
        else:
            some_nonzero += 1
            if len(detail_examples) < 10:
                detail_examples.append({
                    "logId": row["logId"],
                    "direction": direction,
                    "grid_shape": (gh, gw),
                    "obj_coords": obj_coords[:10],
                    "src_values": src_values[:10],
                    "dst_values": dst_values[:10],
                    "grid": g.tolist(),
                })

print(f"\nInterior Move no-ops (first 2000 trajectories): {total_interior_noop}")
print(f"  All selected cells = 0:                    {all_zero_src} ({all_zero_src/total_interior_noop*100:.1f}%)")
print(f"    (src=0 AND dst=0):                       {all_zero_src_and_dst}")
print(f"  Some selected cells non-zero (unexpected): {some_nonzero} ({some_nonzero/total_interior_noop*100:.1f}%)")

if detail_examples:
    print(f"\n--- Non-zero interior no-op examples (Move should have changed grid?) ---")
    for ex in detail_examples[:5]:
        print(f"\n  logId={ex['logId']}, dir={ex['direction']}, grid={ex['grid_shape']}")
        print(f"  Selected coords (y,x): {ex['obj_coords']}")
        print(f"  Source values: {ex['src_values']}")
        print(f"  Destination values: {ex['dst_values']}")
        g = np.array(ex['grid'])
        for r in range(g.shape[0]):
            print(f"    {g[r].tolist()}")
