"""Check: does O2ARC clip objects when moving at the grid boundary?
Focus on Move actions that have selected objects near edges."""

import json
import numpy as np
import pandas as pd

print("Loading ARCTraj CSV...")
df = pd.read_csv("/home/sejin/IntentionLearning/dataset/ARCTraj/ARCTraj.csv")

# Categories of Move behavior
move_empty_sel = 0      # No object selected
move_noop_with_sel = 0  # Object selected but grid unchanged
move_changed = 0        # Grid changed
move_clipped = 0        # Object at edge, moved toward edge, cells disappeared

edge_clip_examples = []
noop_with_sel_examples = []

for row_idx, row in df.iterrows():
    if row_idx > 2000:  # Check first 2000 trajectories
        break

    actions = json.loads(row["actionSequence"])

    for i, a in enumerate(actions):
        if a.get("operation") != "Move":
            continue

        direction = a.get("direction", "")
        objects = a.get("object", [])
        g = np.array(a.get("grid", [[]]), dtype=np.int32)
        gh, gw = g.shape

        # Get next action's grid (= grid after this Move, since grid is PRE-action)
        if i + 1 >= len(actions):
            continue
        ng = np.array(actions[i + 1].get("grid", [[]]), dtype=np.int32)
        if g.shape != ng.shape:
            continue

        has_selection = len(objects) > 0

        if not has_selection:
            move_empty_sel += 1
            continue

        grid_changed = not np.array_equal(g, ng)

        if not grid_changed:
            move_noop_with_sel += 1
            if len(noop_with_sel_examples) < 3:
                # Check if object is at edge
                obj_ys = [o.get("y") for o in objects if isinstance(o.get("y"), (int, float))]
                obj_xs = [o.get("x") for o in objects if isinstance(o.get("x"), (int, float))]
                if obj_ys and obj_xs:
                    noop_with_sel_examples.append({
                        "logId": row["logId"],
                        "direction": direction,
                        "obj_y_range": (min(obj_ys), max(obj_ys)),
                        "obj_x_range": (min(obj_xs), max(obj_xs)),
                        "grid_shape": (gh, gw),
                        "num_cells": len(objects),
                    })
            continue

        # Grid changed - check if it's a clip (object cells disappeared)
        move_changed += 1

        # Analyze: did cells disappear (potential clipping)?
        obj_coords = set()
        for o in objects:
            x, y = o.get("x"), o.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                obj_coords.add((int(y), int(x)))

        # Check if selected object is at the edge in the move direction
        obj_ys = [c[0] for c in obj_coords]
        obj_xs = [c[1] for c in obj_coords]

        if not obj_ys:
            continue

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
            # Check: did the object get clipped?
            # Count non-zero cells in the object region before and after
            diff = g != ng
            cells_that_became_zero = ((g != 0) & (ng == 0) & diff)
            cells_that_became_nonzero = ((g == 0) & (ng != 0) & diff)

            # Object cells that existed before but are now 0
            obj_cells_zeroed = 0
            for (y, x) in obj_coords:
                if 0 <= y < gh and 0 <= x < gw:
                    if g[y, x] != 0 and ng[y, x] == 0:
                        obj_cells_zeroed += 1

            if obj_cells_zeroed > 0:
                move_clipped += 1
                if len(edge_clip_examples) < 5:
                    edge_clip_examples.append({
                        "logId": row["logId"],
                        "direction": direction,
                        "grid_shape": (gh, gw),
                        "num_obj_cells": len(obj_coords),
                        "obj_cells_zeroed": obj_cells_zeroed,
                        "obj_y_range": (min(obj_ys), max(obj_ys)),
                        "obj_x_range": (min(obj_xs), max(obj_xs)),
                        "grid_before": g.tolist(),
                        "grid_after": ng.tolist(),
                    })

print(f"\nMove action breakdown (first 2000 trajectories):")
print(f"  Empty selection (no object): {move_empty_sel}")
print(f"  Has selection, grid unchanged: {move_noop_with_sel}")
print(f"  Has selection, grid changed: {move_changed}")
print(f"  Edge clip detected: {move_clipped}")

if noop_with_sel_examples:
    print(f"\n--- No-op with selection examples ---")
    for ex in noop_with_sel_examples:
        print(f"  logId={ex['logId']}, dir={ex['direction']}, "
              f"obj_y={ex['obj_y_range']}, obj_x={ex['obj_x_range']}, "
              f"grid={ex['grid_shape']}, cells={ex['num_cells']}")

if edge_clip_examples:
    print(f"\n--- Edge clip examples ---")
    for ex in edge_clip_examples:
        print(f"\n  logId={ex['logId']}, dir={ex['direction']}, "
              f"grid={ex['grid_shape']}, obj_cells={ex['num_obj_cells']}, "
              f"zeroed={ex['obj_cells_zeroed']}")
        print(f"  obj_y_range={ex['obj_y_range']}, obj_x_range={ex['obj_x_range']}")
        g = np.array(ex['grid_before'])
        ng = np.array(ex['grid_after'])
        gh, gw = g.shape
        print(f"  Before:")
        for r in range(gh):
            print(f"    {g[r].tolist()}")
        print(f"  After:")
        for r in range(gh):
            print(f"    {ng[r].tolist()}")
        diff = g != ng
        ys, xs = np.where(diff)
        print(f"  Changed {len(ys)} cells:")
        for j in range(min(15, len(ys))):
            print(f"    ({ys[j]},{xs[j]}): {g[ys[j],xs[j]]} -> {ng[ys[j],xs[j]]}")
else:
    print("\n  No edge clip cases found.")

# Also: check Move actions where object is NOT at edge but grid is unchanged
print(f"\n\n--- Extra check: Move with selection, NOT at edge, but no-op ---")
interior_noop = 0
interior_noop_examples = []

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
        if g.shape != ng.shape:
            continue
        if not np.array_equal(g, ng):
            continue
        # It's a no-op with selection. Check if object is NOT at edge.
        obj_ys = [o.get("y") for o in objects if isinstance(o.get("y"), (int, float))]
        obj_xs = [o.get("x") for o in objects if isinstance(o.get("x"), (int, float))]
        if not obj_ys:
            continue
        at_edge = False
        if direction == "up" and min(obj_ys) == 0:
            at_edge = True
        elif direction == "down" and max(obj_ys) == gh - 1:
            at_edge = True
        elif direction == "left" and min(obj_xs) == 0:
            at_edge = True
        elif direction == "right" and max(obj_xs) == gw - 1:
            at_edge = True

        if not at_edge:
            interior_noop += 1
            if len(interior_noop_examples) < 3:
                interior_noop_examples.append({
                    "logId": row["logId"],
                    "direction": direction,
                    "obj_y_range": (min(obj_ys), max(obj_ys)),
                    "obj_x_range": (min(obj_xs), max(obj_xs)),
                    "grid_shape": (gh, gw),
                    "num_cells": len(objects),
                    "grid": g.tolist(),
                })

print(f"  Interior no-op count: {interior_noop}")
for ex in interior_noop_examples:
    print(f"\n  logId={ex['logId']}, dir={ex['direction']}, "
          f"obj_y={ex['obj_y_range']}, obj_x={ex['obj_x_range']}, "
          f"grid={ex['grid_shape']}, cells={ex['num_cells']}")
    g = np.array(ex['grid'])
    for r in range(g.shape[0]):
        print(f"    {g[r].tolist()}")
