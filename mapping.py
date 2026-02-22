"""ARCTraj → JaxARC action mapping and conversion logic.

Key insight: ARCTraj's `grid` field is PRE-action (state BEFORE the action).
Therefore:
    state_t     = action[t].grid      (state before action t)
    next_state_t = action[t+1].grid   (state before action t+1 = state after action t)
"""

import numpy as np
from typing import Optional

# ============================================================
# ARCTraj Operation → JaxARC Operation ID mapping
# ============================================================
# JaxARC operations (0-34):
#   FILL_0..9 (0-9), FLOOD_FILL_0..9 (10-19),
#   MOVE_UP(20), MOVE_DOWN(21), MOVE_LEFT(22), MOVE_RIGHT(23),
#   ROTATE_C(24), ROTATE_CC(25), FLIP_H(26), FLIP_V(27),
#   COPY(28), PASTE(29), CUT(30),
#   CLEAR(31), COPY_INPUT(32), RESIZE(33), SUBMIT(34)

MOVE_DIR_MAP = {
    "up": 20,
    "down": 21,
    "left": 22,
    "right": 23,
}

ROTATE_DIR_MAP = {
    "clockwise": 24,
    "counterclockwise": 25,
}

FLIP_DIR_MAP = {
    "horizontal": 26,
    "vertical": 27,
}


def map_operation(action: dict) -> Optional[int]:
    """Map an ARCTraj action to a JaxARC operation ID.

    Returns None for selection-only actions and Undo/Redo.
    """
    op = action.get("operation")
    cat = action.get("category")

    if cat == "Selection":
        return None
    if cat == "History":
        return None

    if op == "Paint":
        return int(action.get("color", 0))
    if op == "Move":
        return MOVE_DIR_MAP.get(action.get("direction", "up"), 20)
    if op == "Rotate":
        return ROTATE_DIR_MAP.get(action.get("direction", "clockwise"), 24)
    if op == "Flip":
        return FLIP_DIR_MAP.get(action.get("direction", "horizontal"), 26)
    if op == "Copy":
        return 28
    if op == "Paste":
        return 29
    if op == "Submit":
        return 34
    if op == "ResizeGrid":
        return 33

    return None


def _safe_coord(obj: dict) -> tuple[int | None, int | None]:
    """Safely extract (x, y) from an object dict."""
    x = obj.get("x")
    y = obj.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None, None
    return int(x), int(y)


def _set_mask_from_objects(mask: np.ndarray, objects: list[dict], grid_h: int, grid_w: int):
    """Set mask cells to True for each valid object coordinate."""
    for obj in objects:
        x, y = _safe_coord(obj)
        if x is not None and 0 <= y < grid_h and 0 <= x < grid_w:
            mask[y, x] = True


def build_selection_mask(action: dict, grid_h: int, grid_w: int) -> np.ndarray:
    """Build a boolean selection mask from an ARCTraj action."""
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    op = action.get("operation")

    if op == "SelectCell":
        pos = action.get("position", {})
        if pos:
            _set_mask_from_objects(mask, [pos], grid_h, grid_w)

    elif op in ("SelectGrid", "SelectObject"):
        _set_mask_from_objects(mask, action.get("object", []), grid_h, grid_w)

    elif op == "Paint":
        _set_mask_from_objects(mask, action.get("object", []), grid_h, grid_w)

    elif op == "Paste":
        _set_mask_from_objects(mask, action.get("object", []), grid_h, grid_w)

    elif op == "Copy":
        clipboard = action.get("special", {}).get("clipboard", [])
        _set_mask_from_objects(mask, clipboard, grid_h, grid_w)

    elif op in ("Move", "Rotate", "Flip"):
        _set_mask_from_objects(mask, action.get("object", []), grid_h, grid_w)

    elif op in ("Submit", "ResizeGrid"):
        mask[:] = True

    return mask


def _get_grid(action: dict) -> np.ndarray:
    """Extract grid from action as numpy array."""
    g = action.get("grid", [[]])
    return np.array(g, dtype=np.int32)


def convert_trajectory(actions: list[dict], target_grid: np.ndarray = None) -> list[dict]:
    """Convert an ARCTraj action sequence to JaxARC-format transitions.

    Key: ARCTraj grid is PRE-action (state before the action applies).

    Important: O2ARC SelectCell actions can also modify the grid (e.g., clearing
    selected cells). To ensure state continuity, we set each operation's
    next_state = the NEXT OPERATION's PRE-action grid (not the immediately next
    action's grid). This bundles intermediate selection-related grid changes into
    the transition, guaranteeing next_state[t] == state[t+1].

    For the last operation (Submit), next_state = target_grid if provided.

    Args:
        actions: Raw ARCTraj action sequence.
        target_grid: ARC ground truth output (used as next_state for Submit).

    Returns:
        List of transition dicts with dynamic grid sizes per step.
    """
    if not actions:
        return []

    # Step 1: Remove Undo/Redo actions.
    clean = []
    for a in actions:
        if a.get("category") == "History":
            continue
        clean.append(a)

    if not clean:
        return []

    # Step 2: Find all operation indices and collect their metadata.
    op_indices = []
    accumulated_selections = []
    current_acc_sel = None

    for idx, action in enumerate(clean):
        op_id = map_operation(action)
        grid = _get_grid(action)
        gh, gw = grid.shape

        if op_id is None:
            # Selection → accumulate
            if current_acc_sel is None or current_acc_sel.shape != (gh, gw):
                current_acc_sel = np.zeros((gh, gw), dtype=bool)
            sel = build_selection_mask(action, gh, gw)
            current_acc_sel |= sel
        else:
            op_indices.append(idx)
            accumulated_selections.append(
                current_acc_sel.copy() if current_acc_sel is not None else None
            )
            current_acc_sel = None

    if not op_indices:
        return []

    # Step 3: Build transitions.
    # For operation i: state = op_i.grid, next_state = op_{i+1}.grid
    # For last operation: next_state = target_grid
    transitions = []

    for i, idx in enumerate(op_indices):
        action = clean[idx]
        op_id = map_operation(action)
        state = _get_grid(action)
        sh, sw = state.shape

        # next_state = next operation's grid (ensures continuity)
        if i + 1 < len(op_indices):
            next_idx = op_indices[i + 1]
            next_state = _get_grid(clean[next_idx])
        elif target_grid is not None:
            next_state = target_grid
        else:
            continue

        nh, nw = next_state.shape
        acc_sel = accumulated_selections[i]

        # Build selection mask
        # Priority: accumulated selection > operation's own objects > grid diff > full grid
        if acc_sel is not None and acc_sel.shape == (sh, sw) and acc_sel.any():
            final_selection = acc_sel.copy()
            op_sel = build_selection_mask(action, sh, sw)
            if op_sel.any():
                final_selection |= op_sel
        else:
            op_sel = build_selection_mask(action, sh, sw)
            if op_sel.any():
                final_selection = op_sel
            else:
                # Infer selection from grid diff
                if state.shape == next_state.shape:
                    diff_mask = state != next_state
                    if diff_mask.any():
                        final_selection = diff_mask
                    else:
                        final_selection = np.ones((sh, sw), dtype=bool)
                else:
                    final_selection = np.ones((sh, sw), dtype=bool)

        transitions.append({
            "state": state,
            "action_op": op_id,
            "selection": final_selection,
            "next_state": next_state,
            "grid_h": sh,
            "grid_w": sw,
            "next_grid_h": nh,
            "next_grid_w": nw,
        })

    return transitions
