#!/usr/bin/env python3
"""
Streaming pen-stroke animation - single-image render entry point

Renders a colour image into a whiteboard animation in which the pen tip glides along a continuous trajectory, laying down ink as it goes.
The animation has three phases:
  ink pass (ink)   the pen tip lays down black line art along the ink stream
  colour pass (color) the pen retraces the same trajectory, switching to the original colours to light up the picture
  gaze (gaze)  after the pen is lifted, hold on the complete original image

Unlike the 'cell-by-cell jump' approach, this renderer treats the drawing order as the pen tip's polyline path,
interpolates between adjacent landing points, and lets the ink brush lay ink continuously as the tip slides, producing a coherent stream of strokes.
"""
from __future__ import annotations

import argparse
import datetime
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────
# Resource lookup
# ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ASSETS_DIR = _SCRIPT_DIR.parent / "assets"
DEFAULT_HAND_PNG = _ASSETS_DIR / "drawing-hand.png"


def _imread_any(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """
    Read an image, supporting Windows paths that contain non-ASCII characters (e.g. Chinese) or spaces.
    Read the bytes with np.fromfile first, then decode them with cv2.imdecode,
    which sidesteps cv2.imread's compatibility problems with non-ASCII paths.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


# ──────────────────────────────────────────────────────────────
# Render parameters (all in one place)
# ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    fps: int = 60                  # high frame rate makes pen movement look closer to continuous writing
    grid_edge: int = 10            # smaller grid reduces the blocky feel of the line-art reveal
    sample_step: int = 2           # pixel sampling spacing along the pen-tip trajectory
    cap_long_edge: int = 1080      # cap on the long edge of the input image
    brush_radius: int = 40         # radius of the circular ink brush in the colour pass
    ink_weight: int = 2            # line-art phase weight: leaves more time to watch the strokes
    color_weight: int = 1          # colour-pass weight
    gaze_seconds: float = 3.0      # baseline seconds for the gaze phase
    ink_threshold: int = 10        # pixels with grey level below this value count as 'ink'
    ink_reveal_radius: int = 4     # radius of line art the pen tip can reveal per trajectory segment
    target_hand_height: int = 493  # target height of the hand asset after scaling (tuned for 1080p)
    # Normalised coordinates (0..1) of the pen tip within the asset; they decide which asset pixel is aligned to the ink landing point.
    # After cropping, the built-in drawing-hand.png has its pen tip at the top-left corner of the image, so the anchor is (0, 0).
    # This describes the real ink contact point, not the bounding box of the hand image.
    tip_anchor_x: float = 0.0
    tip_anchor_y: float = 0.0
    canvas_hex: str = "#F6F1E3"    # canvas background colour
    match_bg: bool = True          # tint the source-image background to the canvas colour so the ink and colour passes share one background
    match_bg_threshold: int = 28   # pixels whose difference from the source background colour is below this count as background (sum over BGR channels)
    steps_per_frame: int = 4       # baseline number of landing points advanced per frame
    # ── contour-wipe colouring mode only ──
    color_fill: str = "contour-wipe"  # colouring style: "contour-wipe" contour-aware top-to-bottom sweep (default) | "brush" paint along the trajectory
    wipe_decay: float = 0.86       # resistance-field per-row downward decay factor (half-life about 4.6px)
    wipe_delay_ratio: float = 0.04  # fraction of pixels deducted from the front at contours (times h, clamped to [12,52])
    wipe_blocks: int = 18          # number of back-and-forth horizontal sweeps of the pen tip
    # ── Adaptive pauses in the ink pass (simulating a 'change-pen breathing' rhythm) ──
    # pause_mode: "heavy" pronounced pauses (default); "auto" picks a tier by content density; "off" disables pauses; "light" a few pauses
    pause_mode: str = "heavy"
    pause_ratio_heavy: float = 0.03   # pause ratio for low density (slow rhythm): about 3% of frames are pauses
    pause_ratio_light: float = 0.008  # pause ratio for medium density: about 0.8%
    # Density tier thresholds: "frames per cell" (frames_per_cell) measures how much spare animation time there is relative to the content.
    # if >= heavy_fpc there is plenty of spare time -> heavy tier (many pauses); if >= light_fpc moderate -> light tier;
    # if < light_fpc the content is dense and time is tight -> no pauses.
    pause_heavy_fpc: float = 0.7
    pause_light_fpc: float = 0.4
    # ── Stroke path mode ──
    # ink_path_mode: "grid" interpolation through grid-cell centres (default) | "skeleton" skeleton-level pixel tracing
    ink_path_mode: str = "grid"
    skeleton_min_points: int = 8        # minimum points per skeleton stroke (filters fragments)
    skeleton_resample_spacing: float = 2.5  # skeleton resampling spacing (pixels)


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────
def _hex_to_bgr(hex_color: str) -> np.ndarray:
    digits = hex_color.lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"Invalid colour value: {hex_color}")
    r = int(digits[0:2], 16)
    g = int(digits[2:4], 16)
    b = int(digits[4:6], 16)
    return np.array([b, g, r], dtype=np.uint8)


def _bounding_box(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0), (0, 0)
    return (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max()))


# ──────────────────────────────────────────────────────────────
# Ink-cell tiling
# ──────────────────────────────────────────────────────────────
def _to_grid_blocks(image: np.ndarray, edge: int) -> np.ndarray:
    """Split an HxW (x C) image into a tiled view of shape (rows, cols, edge, edge[, C])."""
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    if h % edge or w % edge:
        raise ValueError(f"Image size {w}x{h} must be an integer multiple of {edge}")
    rows, cols = h // edge, w // edge
    if image.ndim == 2:
        return image.reshape(rows, edge, cols, edge).transpose(0, 2, 1, 3)
    return image.reshape(rows, edge, cols, edge, image.shape[2]).transpose(0, 2, 1, 3, 4)


def _active_mask(threshold_map: np.ndarray, edge: int, threshold: int) -> np.ndarray:
    """Which grid cells contain ink: true if the block has any pixel with grey level below the threshold."""
    blocks = _to_grid_blocks(threshold_map, edge)
    return np.any(blocks < threshold, axis=(2, 3))


# ──────────────────────────────────────────────────────────────
# Ink-stream clustering + density-gradient walk
# ──────────────────────────────────────────────────────────────
def _label_components(active: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the ink cells into 8-connected components; returns (label map, component count)."""
    n, labels = cv2.connectedComponents(active.astype(np.uint8), connectivity=8)
    return labels, n - 1  # drop background label 0


def _component_cells(labels: np.ndarray, label: int) -> list[tuple[int, int]]:
    coords = np.argwhere(labels == label)
    return [(int(r), int(c)) for r, c in coords]


def _merge_small_components(
    components: list[list[tuple[int, int]]],
    merge_threshold: int,
) -> list[list[tuple[int, int]]]:
    """
    Merge small connected components (cell count <= merge_threshold) into the spatially nearest large component.
    This avoids many 1-2 cell fragments scattered among large blocks of text, which would make the pen 'jump away before finishing a block of text'.
    If there is no large component to merge into, leave them as they are (no ink is discarded).
    """
    if not components:
        return components
    big = [c for c in components if len(c) > merge_threshold]
    small = [c for c in components if len(c) <= merge_threshold]
    if not small or not big:
        return components

    # precompute the centroid of each large region
    centroids = []
    for cells in big:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        centroids.append((sum(rs) / len(rs), sum(cs) / len(cs)))

    # merge each small fragment into the nearest large region
    merged = [list(cells) for cells in big]  # copy, so it can be appended to
    for cells in small:
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        cr = sum(rs) / len(rs)
        cc = sum(cs) / len(cs)
        best = min(
            range(len(big)),
            key=lambda i: (centroids[i][0] - cr) ** 2 + (centroids[i][1] - cc) ** 2,
        )
        merged[best].extend(cells)
    return merged


def _bounds(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _split_bridge_connected_component(
    cells: list[tuple[int, int]],
    min_side_cells: int = 20,
) -> list[list[tuple[int, int]]]:
    """Split a very wide component when it is connected only by a thin bridge.

    A baseline, arrow, or stray outline can join separate objects into one
    connected component.  Drawing that component with one nearest-neighbour
    walk makes the pen alternate between those objects.  Valleys in the
    vertical ink projection are reliable weak-bridge signals at grid scale.
    """
    if len(cells) < min_side_cells * 2:
        return [cells]

    min_row, min_col, max_row, max_col = _bounds(cells)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if width < 16 or height < 10:
        return [cells]

    counts = {col: 0 for col in range(min_col, max_col + 1)}
    for _, col in cells:
        counts[col] += 1
    valley_limit = max(3, int(np.ceil(height * 0.30)))
    edge_guard = 4
    valleys: list[tuple[int, int]] = []
    start: int | None = None
    for col in range(min_col, max_col + 2):
        low = col <= max_col and counts[col] <= valley_limit
        if low and start is None:
            start = col
        elif not low and start is not None:
            end = col - 1
            if (
                end - start + 1 >= 2
                and start > min_col + edge_guard
                and end < max_col - edge_guard
            ):
                valleys.append((start, end))
            start = None
    if not valleys:
        return [cells]

    # Prefer the broadest empty corridor.  It is much less likely to be an
    # internal detail of a character than a one-column dip.
    start, end = max(valleys, key=lambda band: (band[1] - band[0], -band[0]))
    cut = (start + end) // 2
    left = [cell for cell in cells if cell[1] <= cut]
    right = [cell for cell in cells if cell[1] > cut]
    if len(left) < min_side_cells or len(right) < min_side_cells:
        return [cells]
    return (
        _split_bridge_connected_component(left, min_side_cells)
        + _split_bridge_connected_component(right, min_side_cells)
    )


def _split_bridge_connected_components(
    components: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    return [
        piece
        for cells in components
        for piece in _split_bridge_connected_component(cells)
    ]


def _boxes_touch(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    margin: int = 2,
) -> bool:
    """Whether two component boxes belong to the same visual region."""
    a_top, a_left, a_bottom, a_right = first
    b_top, b_left, b_bottom, b_right = second
    return not (
        a_right + margin < b_left
        or b_right + margin < a_left
        or a_bottom + margin < b_top
        or b_bottom + margin < a_top
    )


def _group_adjacent_stroke_groups(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[list[tuple[str, list[tuple[int, int]]]]]:
    """Keep overlapping label parts and outline pieces in one draw region."""
    regions: list[list[tuple[str, list[tuple[int, int]]]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        group_box = _bounds(group[1])
        touching = [index for index, box in enumerate(boxes) if _boxes_touch(group_box, box)]
        if not touching:
            regions.append([group])
            boxes.append(group_box)
            continue
        target = touching[0]
        regions[target].append(group)
        top, left, bottom, right = boxes[target]
        boxes[target] = (
            min(top, group_box[0]), min(left, group_box[1]),
            max(bottom, group_box[2]), max(right, group_box[3]),
        )
        # Merge any regions newly bridged by the expanded box.
        for index in reversed(touching[1:]):
            regions[target].extend(regions.pop(index))
            other = boxes.pop(index)
            top, left, bottom, right = boxes[target]
            boxes[target] = (
                min(top, other[0]), min(left, other[1]),
                max(bottom, other[2]), max(right, other[3]),
            )
    return regions


def classify_stroke_groups(
    active: np.ndarray,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Classify connected ink regions as a main subject, text, or local contour."""
    labels, count = _label_components(active)
    components = [
        _component_cells(labels, label)
        for label in range(1, count + 1)
    ]
    components = [cells for cells in components if cells]
    if not components:
        return []

    # A long ground line may connect a mountain, a character, and a crowd.
    # Split that weak connection before any region ordering is decided.
    components = _split_bridge_connected_components(components)

    # merge small fragments into the nearest large region so fragments do not interrupt the continuous drawing of large text blocks
    total_cells = sum(len(c) for c in components)
    merge_threshold = max(3, int(total_cells * 0.005))
    components = _merge_small_components(components, merge_threshold)

    subject_index = max(range(len(components)), key=lambda index: len(components[index]))
    groups: list[tuple[str, list[tuple[int, int]], tuple[int, int, int]]] = []
    for index, cells in enumerate(components):
        min_row, min_col, max_row, max_col = _bounds(cells)
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        density = len(cells) / (height * width)
        if index == subject_index:
            kind, rank = "subject", 0
        elif height >= 2 and width / height >= 2.2 and density >= 0.5:
            kind, rank = "text", 1
        else:
            kind, rank = "contour", 2
        groups.append((kind, cells, (rank, min_row, min_col)))

    groups.sort(key=lambda group: group[2])
    return [(kind, cells) for kind, cells, _ in groups]


def _density_seed(cells: Sequence[tuple[int, int]], radius: int = 2) -> tuple[int, int]:
    """Pick the cell with the densest local neighbourhood as the starting point, simulating 'putting pen down where the ink is thickest'."""
    cell_set = set(cells)
    best = cells[0]
    best_score = -1
    for (r, c) in cells:
        score = sum(
            1
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if (r + dr, c + dc) in cell_set
        )
        if score > best_score:
            best_score = score
            best = (r, c)
    return best


def _gradient_walk(cells: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Density-gradient-guided greedy walk: starting from the highest-density seed cell, each step picks
    the unvisited neighbour with the highest local density and the smallest angle to the incoming direction,
    producing a continuous stroke that follows the ink and doubles back as little as possible.
    When no neighbour is reachable, jump to the globally nearest unvisited cell and continue.
    """
    if not cells:
        return []

    cell_set = set(cells)
    seed = _density_seed(cells)
    visited: set[tuple[int, int]] = {seed}
    path: list[tuple[int, int]] = [seed]
    current = seed
    prev_dir = (0, 0)

    while len(visited) < len(cells):
        neighbors = [
            (r, c)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc)
            and (r := current[0] + dr, c := current[1] + dc) in cell_set
            and (r, c) not in visited
        ]
        if neighbors:
            def cost(cell: tuple[int, int]) -> tuple:
                # more neighbours is better (negated so min wins), smaller direction change is better, finally a stable sort by position
                local = sum(
                    1
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                    if (cell[0] + dr, cell[1] + dc) in cell_set
                    and (cell[0] + dr, cell[1] + dc) not in visited
                )
                step = (cell[0] - current[0], cell[1] - current[1])
                turn = (step[0] - prev_dir[0]) ** 2 + (step[1] - prev_dir[1]) ** 2
                return (-local, turn, cell[0], cell[1])

            nxt = min(neighbors, key=cost)
        else:
            # pen break: jump to the nearest unvisited cell
            unvisited = [cell for cell in cells if cell not in visited]
            nxt = min(
                unvisited,
                key=lambda cell: (
                    (cell[0] - current[0]) ** 2 + (cell[1] - current[1]) ** 2,
                    cell[0],
                    cell[1],
                ),
            )

        prev_dir = (nxt[0] - current[0], nxt[1] - current[1])
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path


def _nearest_neighbor_order(
    cells: Sequence[tuple[int, int]], seed: tuple[int, int]
) -> list[tuple[int, int]]:
    """Starting from seed, each step moves to the nearest unvisited cell, forming a continuous stroke."""
    if not cells:
        return []
    remaining = list(cells)
    ordered: list[tuple[int, int]] = []
    current = seed if seed in remaining else remaining[0]
    while remaining:
        ordered.append(current)
        remaining.remove(current)
        if not remaining:
            break
        current = min(
            remaining,
            key=lambda cell: (cell[0] - ordered[-1][0]) ** 2
            + (cell[1] - ordered[-1][1]) ** 2,
        )
    return ordered


def _text_scan_order(
    cells: Sequence[tuple[int, int]], segment_cols: int = 4
) -> list[tuple[int, int]]:
    """
    Dedicated drawing method for text regions: sweep horizontally by segment, simulating writing.
    Split the cells by column into segments (each segment_cols columns wide); segments go left to right;
    within a segment, walk continuously along the ink by nearest neighbour (rather than a raster row-by-row scan), avoiding the feeling of going back to fill in gaps, where a block is left half drawn
    and the next segment starts again from the top.
    """
    if not cells:
        return []
    if segment_cols < 1:
        segment_cols = 1
    left_col = min(col for _, col in cells)
    # bucket by 'start column // segment_cols'; lower bucket numbers (further left) are drawn first
    buckets: dict[int, list[tuple[int, int]]] = {}
    for cell in cells:
        bucket_key = (cell[1] - left_col) // segment_cols
        buckets.setdefault(bucket_key, []).append(cell)

    ordered: list[tuple[int, int]] = []
    prev_tail: tuple[int, int] | None = None
    for key in sorted(buckets):
        seg_cells = buckets[key]
        # segment start: as close as possible to the previous segment's exit, to reduce pen jumps between segments
        if prev_tail is not None:
            seed = min(
                seg_cells,
                key=lambda cell: (cell[0] - prev_tail[0]) ** 2
                + (cell[1] - prev_tail[1]) ** 2,
            )
        else:
            seed = min(seg_cells, key=lambda cell: (cell[0], cell[1]))
        seg_order = _nearest_neighbor_order(seg_cells, seed)
        ordered.extend(seg_order)
        prev_tail = seg_order[-1]
    return ordered


def _order_stream_by_kind(
    kind: str, cells: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Pick the drawing method by region type: text is swept horizontally by segment; subject/contour use the density walk."""
    if kind == "text":
        return _text_scan_order(cells)
    return _gradient_walk(cells)


def _chain_region_paths(
    groups: list[tuple[str, list[tuple[int, int]]]],
) -> list[tuple[int, int]]:
    """Finish every component in one visual region before leaving it."""
    paths = [_order_stream_by_kind(kind, cells) for kind, cells in groups]
    remaining = [path for path in paths if path]
    ordered: list[tuple[int, int]] = []
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_index = 0  # groups retain subject/text/contour priority.
        else:
            pick_index = min(
                range(len(remaining)),
                key=lambda index: min(
                    (remaining[index][0][0] - tail[0]) ** 2
                    + (remaining[index][0][1] - tail[1]) ** 2,
                    (remaining[index][-1][0] - tail[0]) ** 2
                    + (remaining[index][-1][1] - tail[1]) ** 2,
                ),
            )
        path = remaining.pop(pick_index)
        if tail is not None and len(path) > 1:
            head_distance = (path[0][0] - tail[0]) ** 2 + (path[0][1] - tail[1]) ** 2
            end_distance = (path[-1][0] - tail[0]) ** 2 + (path[-1][1] - tail[1]) ** 2
            if end_distance < head_distance:
                path.reverse()
        ordered.extend(path)
        tail = path[-1]
    return ordered


def cluster_ink_streams(active: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Group the ink cells semantically into several ink streams: subject -> text -> local contour,
    each stream picks a drawing method by type (text swept by segment, the rest density walk);
    streams are chained dynamically by nearest 'exit to entry' distance, reversing a whole stream when needed to cut down on pen jumps.
    Returns the chained and ordered list of stroke streams.
    """
    if not active.any():
        return []
    groups = classify_stroke_groups(active)
    # A stream is now a complete visual region, not merely one connected
    # component.  Thus a label's border, its characters, and its arrow cannot
    # be interrupted by a different object that happens to be closer.
    regions = _group_adjacent_stroke_groups(groups)
    streams = [_chain_region_paths(region) for region in regions]
    streams = [s for s in streams if s]
    if not streams:
        return []

    # Chaining: the subject (first stream) goes first; after that, each time pick the stream whose entry is nearest the current exit,
    # and reverse that stream as a whole if useful so its start is closer to the previous stream's exit.
    ordered: list[list[tuple[int, int]]] = []
    remaining = list(streams)
    tail: tuple[int, int] | None = None
    while remaining:
        if tail is None:
            pick_idx = 0  # classify already put the subject first
        else:
            def dist_to_tail(stream: list[tuple[int, int]]) -> int:
                head = stream[0]
                return (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            pick_idx = min(range(len(remaining)), key=lambda i: dist_to_tail(remaining[i]))
        pick = remaining.pop(pick_idx)
        # reverse if useful: if the tail is closer to pick's end than to its start, reverse
        if tail is not None and len(pick) > 1:
            head = pick[0]
            end = pick[-1]
            d_end = (end[0] - tail[0]) ** 2 + (end[1] - tail[1]) ** 2
            d_head = (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
            if d_end < d_head:
                pick = pick[::-1]
        ordered.append(pick)
        tail = pick[-1]
    return ordered


def flatten_streams(streams: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for stream in streams for cell in stream]


# ──────────────────────────────────────────────────────────────
# Pen tip / hand overlay
# ──────────────────────────────────────────────────────────────
def _load_hand(path: Path, target_h: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load the hand asset and scale it proportionally to the target height.
    Prefer the alpha channel as the mask; without alpha, fall back to 'near-white is background' detection.
    Returns (hand BGR, normalised mask [0..1]); returns None on failure.
    """
    if not path.exists():
        return None
    raw = _imread_any(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None

    if raw.ndim == 3 and raw.shape[2] == 4:
        hand = raw[:, :, :3]
        mask = raw[:, :, 3]
    else:
        hand = raw
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # crop to the valid area
    (x0, y0), (x1, y1) = _bounding_box(mask)
    if x1 <= x0 or y1 <= y0:
        return None
    hand = hand[y0:y1 + 1, x0:x1 + 1]
    mask = mask[y0:y1 + 1, x0:x1 + 1]

    scale = target_h / hand.shape[0]
    new_w = max(1, int(round(hand.shape[1] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    hand = cv2.resize(hand, (new_w, target_h), interpolation=interp)
    mask = cv2.resize(mask, (new_w, target_h), interpolation=interp)
    mask = mask.astype(np.float32) / 255.0

    # blacken everything outside the mask to simplify later mask blending
    hand[mask <= 0] = 0
    return hand, mask


def _procedural_tip(target_h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback pen tip: procedurally draw a marker pen (gradient barrel + round soft-edged head + drop shadow).
    It depends on no external image, so output still works when the asset is missing.
    """
    w = max(1, int(target_h * 0.34))
    h = target_h
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # drop shadow: an offset dark band, softened and placed underneath
    shadow = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(shadow, (3, int(h * 0.06)), (w - 2, int(h * 0.62)), 90, thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (15, 15), 0)
    rgba[:, :, 3] = shadow

    # barrel: vertical gradient from light to dark
    for y in range(h):
        t = y / max(1, h - 1)
        shade = int(220 - 130 * t)
        rgba[y, :, 0:3] = (shade, shade, shade + 10)
    cv2.rectangle(rgba, (4, int(h * 0.04)), (w - 4, int(h * 0.58)), (0, 0, 0), thickness=1)

    # round pen tip (warm colour, simulating ink)
    tip_cy = int(h * 0.70)
    cv2.circle(rgba, (w // 2, tip_cy), max(3, w // 4), (70, 90, 230), thickness=-1)

    # compose the alpha mask from the round head plus the barrel outline
    body_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(body_mask, (3, int(h * 0.04)), (w - 3, tip_cy), 255, thickness=-1)
    cv2.circle(body_mask, (w // 2, tip_cy), max(3, w // 4), 255, thickness=-1)
    body_mask = cv2.GaussianBlur(body_mask, (7, 7), 0)

    hand = rgba[:, :, :3]
    mask = np.maximum(rgba[:, :, 3], body_mask).astype(np.float32) / 255.0
    hand[mask <= 0] = 0
    return hand, mask


class TipOverlay:
    """Paste the pen tip/hand onto the canvas, aligning the specified 'pen-tip anchor' with the ink landing point, with alpha blending."""

    def __init__(
        self,
        hand: np.ndarray,
        mask: np.ndarray,
        tip_anchor_x: float = 0.0,
        tip_anchor_y: float = 0.0,
    ) -> None:
        self.hand = hand
        self.mask = mask
        self.h, self.w = hand.shape[:2]
        self.mask_inv = 1.0 - mask
        # pixel coordinates of the pen tip within the asset (the ink landing point is aligned to it)
        # Map normalized anchors exactly onto the source image's pixel range.
        self.tip_px = int(round((self.w - 1) * np.clip(tip_anchor_x, 0.0, 1.0)))
        self.tip_py = int(round((self.h - 1) * np.clip(tip_anchor_y, 0.0, 1.0)))

    def stamp(self, canvas: np.ndarray, x: int, y: int) -> np.ndarray:
        """Align the asset's pen-tip anchor to canvas coordinates (x, y), i.e. the ink landing point."""
        # asset top-left = ink landing point - pen-tip offset
        anchor_x = x - self.tip_px
        anchor_y = y - self.tip_py
        h_canvas, w_canvas = canvas.shape[:2]

        x0 = max(0, anchor_x)
        y0 = max(0, anchor_y)
        x1 = min(w_canvas, anchor_x + self.w)
        y1 = min(h_canvas, anchor_y + self.h)
        if x1 <= x0 or y1 <= y0:
            return canvas

        sx0 = x0 - anchor_x
        sy0 = y0 - anchor_y
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)

        region = canvas[y0:y1, x0:x1]
        hand_region = self.hand[sy0:sy1, sx0:sx1]
        mask_region = self.mask[sy0:sy1, sx0:sx1]
        inv_region = self.mask_inv[sy0:sy1, sx0:sx1]

        for c in range(3):
            region[:, :, c] = (
                region[:, :, c] * inv_region + hand_region[:, :, c] * mask_region
            )
        canvas[y0:y1, x0:x1] = region
        return canvas


# ──────────────────────────────────────────────────────────────
# Ink brush
# ──────────────────────────────────────────────────────────────
def _feathered_disk(radius: int) -> np.ndarray:
    """Generate a circular mask of radius r with Gaussian-feathered edges, value range 0..1."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    dist = np.sqrt(x * x + y * y).astype(np.float32)
    return np.clip(1.0 - (dist - radius * 0.75) / (radius * 0.25), 0.0, 1.0)


# ──────────────────────────────────────────────────────────────
# contour-wipe colouring helpers
# ──────────────────────────────────────────────────────────────
def _ease_in_out_sine(t: float | np.ndarray) -> float | np.ndarray:
    """Sine easing: slow at the start and end, fast in the middle. Accepts a scalar or array; output has the same shape."""
    return -(np.cos(np.pi * t) - 1.0) / 2.0


def _build_wipe_wave(width: int) -> np.ndarray:
    """
    Precompute a dual-frequency sine-wave boundary so the reveal front is a rippling wave rather than a straight line.
    Returns a (W,) float32 array with a value range of roughly [-1.35, 1.35].
    """
    wave_px1 = max(24.0, width / 20.0)
    wave_px2 = max(8.0, width / 72.0)
    xs = np.arange(width, dtype=np.float32)
    return np.sin(xs / wave_px1) + 0.35 * np.sin(xs / wave_px2 + 1.7)


# ──────────────────────────────────────────────────────────────
# Skeleton-level stroke tracing (ported from whiteboard-video-engine preprocess.py)
# Zhang-Suen thinning -> 8-neighbour straightest-edge tracing -> pixel-level ordered strokes
# ──────────────────────────────────────────────────────────────
_SKEL_NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
]


def _zhang_suen_skeleton(mask: np.ndarray, max_iterations: int = 160) -> np.ndarray:
    """
    Zhang-Suen two-sub-iteration thinning, reducing the binary foreground mask to a 1px-wide skeleton.
    Input: 2-D bool/uint8 array (True/1 = foreground stroke).
    Output: bool skeleton image of the same shape.
    """
    img = np.pad(mask.astype(np.uint8), 1, mode="constant")
    for _ in range(max_iterations):
        changed = False
        for step in (0, 1):
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            center = img[1:-1, 1:-1]
            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            # number of 0->1 transitions (clockwise around the point)
            transitions = sum(
                (neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8)
            )
            count = sum(neighbors)
            if step == 0:
                marker = (
                    (center == 1) & (count >= 2) & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
                )
            else:
                marker = (
                    (center == 1) & (count >= 2) & (count <= 6)
                    & (transitions == 1)
                    & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
                )
            if np.any(marker):
                center[marker] = 0
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(bool)


def _skel_neighbors(skel: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    """
    Return the valid 8-connected neighbours of skeleton point point.
    Key point: when an orthogonal bridge already links a diagonal neighbour to the current point, skip that diagonal neighbour,
    avoiding triangular stroke fragments at T-junctions and crossings while keeping genuine pure-diagonal centre lines.
    """
    x, y = point
    h, w = skel.shape
    result: list[tuple[int, int]] = []
    for dx, dy in _SKEL_NEIGHBORS_8:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < w and 0 <= ny < h and skel[ny, nx]):
            continue
        if dx != 0 and dy != 0 and (skel[y, nx] or skel[ny, x]):
            continue  # orthogonal bridge already connects them; skip the redundant diagonal
        result.append((nx, ny))
    return result


def _edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Undirected edge normalisation: (A,B) and (B,A) map to the same key."""
    return (a, b) if a <= b else (b, a)


def _choose_next(
    prev: tuple[int, int],
    cur: tuple[int, int],
    candidates: list[tuple[int, int]],
    visited_edges: set,
) -> tuple[int, int] | None:
    """
    At a junction, choose the 'straightest unvisited edge' to continue.
    Straightness is measured by the cosine similarity between the current heading and each candidate direction; take the maximum.
    """
    fresh = [p for p in candidates if _edge_key(cur, p) not in visited_edges and p != prev]
    if not fresh:
        return None
    vx, vy = cur[0] - prev[0], cur[1] - prev[1]
    vlen = math.hypot(vx, vy)
    return max(
        fresh,
        key=lambda p: (
            (vx * (p[0] - cur[0]) + vy * (p[1] - cur[1]))
            / (vlen * math.hypot(p[0] - cur[0], p[1] - cur[1]) or 1.0)
        ),
    )


def trace_8connected(skel: np.ndarray, min_points: int = 8) -> list[list[tuple[int, int]]]:
    """
    Trace the 1px skeleton into ordered stroke sequences.

    - Start priority: endpoints with degree 1 -> junctions with degree > 2 -> others
    - At junctions continue along the straightest unvisited edge (instead of breaking every branch into a short stroke)
    - Mark visits with a set of undirected edges (pixels may be reused, edges may not be walked twice)
    - Stop at a dead end (no fresh edge); remaining branches are picked up by later starting points
    - Fragments shorter than min_points are discarded

    Returns list[list[(x,y)]]; each is an ordered list of pixel coordinates along the stroke direction.
    """
    ys, xs = np.nonzero(skel)
    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    if not points:
        return []
    degrees = {p: len(_skel_neighbors(skel, p)) for p in points}
    starts = (
        [p for p in points if degrees[p] == 1]
        + [p for p in points if degrees[p] > 2]
        + points
    )
    visited_edges: set = set()
    strokes: list[list[tuple[int, int]]] = []
    for start in starts:
        for nb in _skel_neighbors(skel, start):
            edge = _edge_key(start, nb)
            if edge in visited_edges:
                continue
            path = [start]
            prev, cur = start, nb
            visited_edges.add(edge)
            while True:
                path.append(cur)
                next_pt = _choose_next(prev, cur, _skel_neighbors(skel, cur), visited_edges)
                if next_pt is None:
                    break
                visited_edges.add(_edge_key(cur, next_pt))
                prev, cur = cur, next_pt
            if len(path) >= min_points:
                strokes.append(path)
    return strokes


# ── Skeleton stroke post-processing (resample + smooth + sort) ──
def _stroke_cumulative_length(points: list[tuple[float, float]]) -> list[float]:
    """Cumulative arc length at each point [0, d01, d012, ...]."""
    cum = [0.0]
    for a, b in zip(points, points[1:]):
        cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    return cum


def _resample_stroke_points(
    points: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """Resample at equal spacing along the arc length to remove pixel jaggedness."""
    if len(points) < 2:
        return list(points)
    cum = _stroke_cumulative_length(points)
    total = cum[-1]
    if total < spacing:
        return [points[0], points[-1]]
    n = max(2, int(round(total / spacing)))
    result: list[tuple[float, float]] = []
    for i in range(n + 1):
        target = total * i / n
        # binary-search locate
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            result.append(points[0])
            continue
        seg_start = cum[lo - 1]
        seg_len = cum[lo] - seg_start
        t = (target - seg_start) / seg_len if seg_len > 0 else 0.0
        ax, ay = points[lo - 1]
        bx, by = points[lo]
        result.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return result


def _chaikin_smooth(
    points: list[tuple[float, float]], iterations: int = 1
) -> list[tuple[float, float]]:
    """Chaikin corner-cutting smoothing: replace each segment with its 0.25/0.75 points, keeping the start and end points."""
    pts = list(points)
    for _ in range(iterations):
        if len(pts) < 3:
            break
        smoothed = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            smoothed.append((a[0] * 0.75 + b[0] * 0.25, a[1] * 0.75 + b[1] * 0.25))
            smoothed.append((a[0] * 0.25 + b[0] * 0.75, a[1] * 0.25 + b[1] * 0.75))
        smoothed.append(pts[-1])
        pts = smoothed
    return pts


def _order_skeleton_strokes(strokes: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """
    Stroke ordering: top to bottom, left to right, longer strokes first.
    Simplified order_strokes: lexicographic order on the bounding-box top-left corner plus negative length.
    """
    def sort_key(s):
        if not s:
            return (0, 0, 0, 0)
        xs = [p[0] for p in s]
        ys = [p[1] for p in s]
        length = _stroke_cumulative_length(s)[-1]
        return (min(ys) // 12, min(xs), min(ys), -length)
    return sorted(strokes, key=sort_key)


# ──────────────────────────────────────────────────────────────
# Duration / phase split
# ──────────────────────────────────────────────────────────────
@dataclass
class PhasePlan:
    ink_frames: int
    color_frames: int
    gaze_frames: int
    ratio_label: str


def plan_phases(total_ms: int, cfg: Config) -> PhasePlan:
    """
    Split the total duration into three phases: ink pass / colour pass / gaze.
    The gaze phase is reserved first using the baseline seconds; the remaining time is split between the ink and colour passes by weight;
    if the remainder is not divisible by the weight sum, the leftover goes to the gaze phase to avoid losing precision.
    """
    weight_sum = cfg.ink_weight + cfg.color_weight
    gaze_ms = int(cfg.gaze_seconds * 1000)
    anim_ms = total_ms - gaze_ms
    remainder = anim_ms % weight_sum
    if remainder:
        anim_ms -= remainder
        gaze_ms += remainder

    ink_frames = round(anim_ms * cfg.ink_weight / weight_sum * cfg.fps / 1000)
    color_frames = round(anim_ms * cfg.color_weight / weight_sum * cfg.fps / 1000)
    gaze_frames = round(gaze_ms * cfg.fps / 1000)
    if ink_frames <= 0 and color_frames <= 0:
        ink_frames = color_frames = 0
    return PhasePlan(ink_frames, color_frames, gaze_frames, f"{cfg.ink_weight}:{cfg.color_weight}")


# ──────────────────────────────────────────────────────────────
# Renderer core
# ──────────────────────────────────────────────────────────────
class StreamBoardRenderer:
    """Holds all state for a single render; methods hang off the instance."""

    def __init__(
        self,
        image_bgr: np.ndarray,
        cfg: Config,
        hand_png: Path | None,
        bare_tip: bool,
    ) -> None:
        self.cfg = cfg
        self.canvas_bgr = _hex_to_bgr(cfg.canvas_hex)

        # Compute the output size: clamp the long edge to cap and align to an even multiple of grid_edge (the encoder requires even dimensions)
        h0, w0 = image_bgr.shape[:2]
        scale = cfg.cap_long_edge / max(h0, w0)
        w = int(round(w0 * scale))
        h = int(round(h0 * scale))
        align = cfg.grid_edge if cfg.grid_edge % 2 == 0 else cfg.grid_edge * 2
        w = (w // align) * align
        h = (h // align) * align
        self.out_w = max(align, w)
        self.out_h = max(align, h)

        self.color_img = cv2.resize(image_bgr, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)
        self.thresh_map = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
        )
        self.active = _active_mask(self.thresh_map, cfg.grid_edge, cfg.ink_threshold)
        self.grid_blocks = _to_grid_blocks(self.thresh_map, cfg.grid_edge)
        self.ink_pixels = self.thresh_map < cfg.ink_threshold
        self.ink_paint = np.repeat(self.thresh_map[:, :, None], 3, axis=2).astype(np.float32)

        # Tint the source-image background to the canvas colour (affects only color_img; ink_pixels / ink_paint are untouched).
        # This keeps the background in the colour and gaze phases consistent with the ink (line-art) phase, avoiding an abrupt background colour jump.
        # It runs after ink_pixels is computed, so line-art quality is completely unaffected.
        if cfg.match_bg:
            self._match_original_background()

        # cell-space clustering (still needed by the grid-mode stroke path and the contour-wipe resistance field)
        self.ink_streams = cluster_ink_streams(self.active)

        # stroke path: choose grid (cell-centre interpolation) or skeleton (skeleton tracing) by ink_path_mode
        if cfg.ink_path_mode == "skeleton":
            self.skeleton_strokes = self._build_skeleton_path()
            if self.skeleton_strokes:
                self.stroke_path = [pt for stroke in self.skeleton_strokes for pt in stroke]
            else:
                # skeleton tracing produced no strokes: genuinely fall back to the cell-centre path rather than leaving an empty path
                self.stroke_path = flatten_streams(self.ink_streams)
        else:
            self.skeleton_strokes = []
            self.stroke_path = flatten_streams(self.ink_streams)

        # canvas (float buffer, to make brush accumulation blending easy)
        self.drawn = np.zeros((self.out_h, self.out_w, 3), dtype=np.float32)
        self.drawn[...] = self.canvas_bgr.astype(np.float32)

        # pen-tip overlay
        self.tip: TipOverlay | None = None
        if not bare_tip:
            hand_data = _load_hand(hand_png, cfg.target_hand_height) if hand_png else None
            tip_anchor_x = cfg.tip_anchor_x
            tip_anchor_y = cfg.tip_anchor_y
            if hand_data is None:
                hand_data = _procedural_tip(cfg.target_hand_height)
                tip_anchor_x = 0.5
                tip_anchor_y = 0.70
            self.tip = TipOverlay(
                hand_data[0], hand_data[1],
                tip_anchor_x=tip_anchor_x,
                tip_anchor_y=tip_anchor_y,
            )

    # ── Tint the source background to the canvas colour (affects only color_img, not the line-art ink) ──
    def _match_original_background(self) -> None:
        """
        Sample the four corners of the source image as the background reference and replace pixels whose difference is < the threshold with canvas_hex.
        This keeps the colour and gaze phases' background consistent with the ink (line-art) phase, avoiding an abrupt background colour jump.
        Coloured content (strongly different from the background) keeps its original colour.
        """
        img = self.color_img
        h, w = img.shape[:2]
        margin = max(3, min(h, w) // 50)
        samples = [
            img[:margin, :margin], img[:margin, -margin:],
            img[-margin:, :margin], img[-margin:, -margin:],
        ]
        bg_color = np.median(np.concatenate([s.reshape(-1, 3) for s in samples]), axis=0)
        diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16)).sum(axis=2)
        bg_mask = diff < self.cfg.match_bg_threshold
        img[bg_mask] = self.canvas_bgr

    # ── Stroke centre points (pixel coordinates) ──
    def _cell_center(self, cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        e = self.cfg.grid_edge
        return (c * e + e // 2, r * e + e // 2)  # (x, y)

    # ── Skeleton-level stroke path (Zhang-Suen thinning + 8-neighbour straightest-edge tracing) ──
    def _build_skeleton_path(self) -> list[list[tuple[int, int]]]:
        """
        Use skeleton tracing to generate pixel-level ordered stroke sequences, replacing grid cell-centre interpolation.
        The pen tip follows the real skeleton, hugging the source lines more closely than grid centres;
        at junctions it continues along the straightest edge, avoiding triangular fragments.
        """
        cfg = self.cfg
        skel = _zhang_suen_skeleton(self.ink_pixels, max_iterations=160)
        raw_strokes = trace_8connected(skel, min_points=cfg.skeleton_min_points)
        if not raw_strokes:
            print("  [warn] skeleton tracing found no strokes; falling back to the cell-centre path")
            return []

        spacing = cfg.skeleton_resample_spacing
        processed: list[list[tuple[int, int]]] = []
        for stroke in raw_strokes:
            pts = [(float(x), float(y)) for x, y in stroke]
            pts = _resample_stroke_points(pts, spacing)
            pts = _chaikin_smooth(pts, iterations=1)
            pts = _resample_stroke_points(pts, spacing)
            if len(pts) >= 2 and _stroke_cumulative_length(pts)[-1] > 2.0:
                processed.append([(int(round(x)), int(round(y))) for x, y in pts])

        processed = _order_skeleton_strokes(processed)
        total_pts = sum(len(s) for s in processed)
        print(f"  skeleton tracing: {len(processed)} strokes, {total_pts} sample points")
        return processed

    # ── contour-wipe resistance field (built lazily, reused throughout the colour pass) ──
    def _build_resistance_field(self) -> np.ndarray:
        """
        Build a 'resistance field' from the line-art ink: resistance is about 1 at contours and decays exponentially by decay per row going downward.
        The reveal front loses pixels when it meets high resistance, so it 'first stalls on the contour, then slowly crosses it'.

        The resistance field is static throughout and independent of colouring progress, so it is computed once and cached in self._resistance.
        """
        if getattr(self, "_resistance", None) is not None:
            return self._resistance

        h, w = self.out_h, self.out_w
        cfg = self.cfg

        # 1) binary ink-line image (uint8 0/255)
        ink_u8 = (self.ink_pixels.astype(np.uint8)) * 255

        # 2) dilation: a circular structuring element thickens contours into a blocking band
        spread = int(np.clip(min(w, h) // 64, 3, 17))
        if spread % 2 == 0:  # the structuring element size must be a positive odd number
            spread = max(3, spread - 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spread, spread))
        dilated = cv2.dilate(ink_u8, kernel, iterations=1)

        # 3) Gaussian blur: turn hard edges into a gradient band (kernel size must be a positive odd number)
        blur_r = max(1, int(round(min(w, h) / 220.0)))
        if blur_r % 2 == 0:
            blur_r += 1
        resistance = cv2.GaussianBlur(dilated, (blur_r, blur_r), 0).astype(np.float32)

        # 4) normalise to [0,1]
        peak = float(resistance.max())
        if peak > 1e-6:
            resistance /= peak
        else:
            # no ink lines (all-white image): the resistance field is constantly 0 and contour-wipe degrades to a straight sweep
            resistance = np.zeros((h, w), dtype=np.float32)

        # 5) causal row-by-row downward propagation of decay: each contour casts an exponentially decaying shadow downward
        decay = cfg.wipe_decay
        for row in range(1, h):
            resistance[row] = np.maximum(resistance[row], resistance[row - 1] * decay)

        self._resistance = resistance
        return resistance

    # ── Lay an 'ink dot' at a landing point: threshold image in the line-art phase, original colours in the colour pass ──
    def _reveal_ink_segment(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> None:
        """Reveal only the original line-art pixels touched by one pen movement."""
        segment = np.zeros((self.out_h, self.out_w), dtype=np.uint8)
        thickness = max(1, self.cfg.ink_reveal_radius * 2 + 1)
        cv2.line(segment, start, end, 255, thickness=thickness, lineType=cv2.LINE_AA)
        revealed = (segment > 0) & self.ink_pixels
        self.drawn[revealed] = self.ink_paint[revealed]

    def _ink_stamp(self, cell: tuple[int, int]) -> None:
        r, c = cell
        e = self.cfg.grid_edge
        block = self.grid_blocks[r, c]
        ink_region = block < self.cfg.ink_threshold
        # the threshold image is single-channel; replicate it to the three-channel canvas
        paint = np.repeat(block[:, :, None], 3, axis=2)
        target = self.drawn[r * e:r * e + e, c * e:c * e + e]
        target[ink_region] = paint[ink_region]

    def _color_stamp(self, px: int, py: int, disk: np.ndarray) -> None:
        radius = self.cfg.brush_radius
        h, w = self.out_h, self.out_w
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        if y1 <= y0 or x1 <= x0:
            return
        by0, by1 = y0 - (py - radius), disk.shape[0] - ((py + radius + 1) - y1)
        bx0, bx1 = x0 - (px - radius), disk.shape[1] - ((px + radius + 1) - x1)
        m = disk[by0:by1, bx0:bx1]
        inv = 1.0 - m
        target = self.drawn[y0:y1, x0:x1]
        source = self.color_img[y0:y1, x0:x1].astype(np.float32)
        for ch in range(3):
            target[:, :, ch] = target[:, :, ch] * inv + source[:, :, ch] * m

    # ── Write the current canvas snapshot (including the pen tip) for several frames ──
    def _snapshot_with_tip(self, px: int, py: int) -> np.ndarray:
        snap = self.drawn.astype(np.uint8)  # astype already returns a new array, no extra copy needed
        if self.tip is not None:
            self.tip.stamp(snap, px, py)
        return snap

    def _build_stroke_samples(
        self, path: list[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], set[int], list[int]]:
        """
        Interpolate the stroke polyline into a continuous sequence of pen-tip pixel coordinates.
        Between adjacent cell centres, sample uniformly every sample_step pixels to form a coherent gliding trajectory.

        Returns (samples, pen_lifts, sample_cell_index):
          samples         —— list of pen-tip pixel coordinates
          pen_lifts       —— set of sample indices where the pen lifts (where it switches between non-adjacent cells)
          sample_cell_index —— index in path of the cell each sample point belongs to,
                              used to keep the ink-reveal progress strictly in sync with the pen-tip position.
        """
        samples: list[tuple[int, int]] = []
        pen_lifts: set[int] = set()
        sample_cell_index: list[int] = []
        for idx, cell in enumerate(path):
            cx, cy = self._cell_center(cell)
            if idx == 0:
                samples.append((cx, cy))
                sample_cell_index.append(idx)
                continue
            prev_cell = path[idx - 1]
            prev = self._cell_center(prev_cell)
            cell_distance = math.hypot(cell[0] - prev_cell[0], cell[1] - prev_cell[1])
            if cell_distance > math.sqrt(2):
                pen_lifts.add(len(samples))
                samples.append((cx, cy))
                sample_cell_index.append(idx)
                continue
            steps = max(
                1, int(math.hypot(cx - prev[0], cy - prev[1]) / self.cfg.sample_step)
            )
            for s in range(1, steps + 1):
                samples.append(
                    (int(prev[0] + (cx - prev[0]) * s / steps),
                     int(prev[1] + (cy - prev[1]) * s / steps))
                )
                sample_cell_index.append(idx)
        return samples, pen_lifts, sample_cell_index

    def _frame_progress_indices(self, n_steps: int, target_frames: int) -> list[int]:
        """
        Given n_steps pen-tip positions and target_frames target frames,
        return the pen-tip position index each target frame should use (uniform mapping covering the full trajectory).
        When target_frames <= n_steps this is downsampling; otherwise it is repeated sampling.
        When target_frames <= 0 (e.g. total duration <= gaze phase leaves this phase no frames) return empty and produce no frames.
        """
        if n_steps == 0 or target_frames <= 0:
            return []
        if target_frames == 1:
            return [n_steps - 1]
        return [
            round(f * (n_steps - 1) / (target_frames - 1))
            for f in range(target_frames)
        ]

    def _pause_frame_indices(
        self, target_frames: int, n_cells: int
    ) -> set[int]:
        """
        Adaptive pauses: pick the pause ratio by content-density tier, then spread the pause frames evenly along the timeline.
        Returns the set of frame indices that should be frozen (repeat the previous frame's progress).

        The tier metric is 'frames per cell', frames_per_cell = target_frames / n_cells:
        far more frames than cells (large value) means there is spare animation time relative to the content -> more pauses to simulate change-pen breathing;
        fewer frames than cells (small value) means dense content and tight timing -> no pauses.

        pause_mode can force an override: "off" disables, "light"/"heavy" fix the tier, "auto" chooses automatically.
        """
        mode = self.cfg.pause_mode
        if mode == "off" or target_frames < 8 or n_cells <= 0:
            return set()

        if mode == "light":
            ratio = self.cfg.pause_ratio_light
        elif mode == "heavy":
            ratio = self.cfg.pause_ratio_heavy
        else:  # auto: choose the tier automatically by 'frames per cell'
            fpc = target_frames / n_cells
            if fpc >= self.cfg.pause_heavy_fpc:
                ratio = self.cfg.pause_ratio_heavy
            elif fpc >= self.cfg.pause_light_fpc:
                ratio = self.cfg.pause_ratio_light
            else:
                return set()  # dense content: fast rhythm, no pauses

        # number of pause frames, at least 0; clamped to target_frames-2 to avoid pausing on the first and last frames
        pause_count = min(
            max(0, int(round(target_frames * ratio))),
            max(0, target_frames - 2),
        )
        if pause_count <= 0:
            return set()

        # equal-spacing insertion: split target_frames into pause_count+1 equal parts and put pauses on the interior split points
        # the first and last are not used (the numerator of 1/(n+1) starts at 1), so the start and end are never interrupted
        return {
            max(1, min(target_frames - 2,
                       round((idx + 1) * target_frames / (pause_count + 1))))
            for idx in range(pause_count)
        }

    # ── Ink pass: lay line art along stroke_path, with the pen tip gliding in strict sync with the ink reveal ──
    def lay_down_ink(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """Ink-pass entry point: dispatch by ink_path_mode to skeleton tracing or the grid-cell path."""
        if self.cfg.ink_path_mode == "skeleton" and self.skeleton_strokes:
            return self._lay_down_ink_skeleton(writer, target_frames)
        return self._lay_down_ink_grid(writer, target_frames)

    # ── grid mode: reveal ink along the interpolated grid-cell-centre path (original logic) ──
    def _lay_down_ink_grid(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        path = self.stroke_path
        n = len(path)
        if n == 0:
            print("  no ink, skipping the ink pass")
            for _ in range(target_frames):
                writer.write(self._snapshot_with_tip(self.out_w // 2, self.out_h // 2))
            return

        samples, pen_lifts, sample_cell_index = self._build_stroke_samples(path)
        sample_idx_for_frame = self._frame_progress_indices(len(samples), target_frames)

        # Adaptive pauses: select 'frozen frames' by density tier (pen tip stays still, ink reveal does not advance),
        # simulating the pen-change/breathing rhythm of real handwriting.
        pause_frames = self._pause_frame_indices(target_frames, n)
        if pause_frames:
            print(f"  adaptive pauses: {len(pause_frames)} frames frozen (mode={self.cfg.pause_mode})")

        written = 0
        cells_revealed = 0  # number of cells revealed as whole blocks (incremental, strictly following pen-tip progress)
        last_sample_idx: int | None = None
        for fi, si in enumerate(sample_idx_for_frame):
            # pause frame: reuse the previous frame's pen-tip position and progress, reveal no ink, just write one snapshot frame (pen tip frozen)
            if fi in pause_frames and last_sample_idx is not None:
                sx, sy = samples[last_sample_idx]
                writer.write(self._snapshot_with_tip(sx, sy))
                written += 1
                if (fi + 1) % max(1, target_frames // 10) == 0:
                    print(f"  ink pass progress: {int((fi + 1) / target_frames * 100)}%")
                continue

            # the pen tip reveals along the line (preserves the flowing feel of the strokes)
            if last_sample_idx is None:
                self._reveal_ink_segment(samples[si], samples[si])
            else:
                for sample_idx in range(last_sample_idx + 1, si + 1):
                    if sample_idx in pen_lifts:
                        continue
                    self._reveal_ink_segment(
                        samples[sample_idx - 1], samples[sample_idx]
                    )

            # block reveal: reveal strictly up to 'the cell the pen tip is currently in', keeping the pen tip and drawing in sync and the text intact.
            # sample_cell_index[si] is the index of the cell the current frame's pen tip belongs to; reveal up to it.
            target_cell = sample_cell_index[si]
            while cells_revealed <= target_cell and cells_revealed < n:
                self._ink_stamp(path[cells_revealed])
                cells_revealed += 1

            sx, sy = samples[si]
            writer.write(self._snapshot_with_tip(sx, sy))
            written += 1
            last_sample_idx = si
            if (fi + 1) % max(1, target_frames // 10) == 0:
                print(f"  ink pass progress: {int((fi + 1) / target_frames * 100)}%")

        # final safety net: make sure all cells' ink is fully revealed and pad out the frame count
        while cells_revealed < n:
            self._ink_stamp(path[cells_revealed])
            cells_revealed += 1
        last = samples[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  ink pass done: {n} cells, {written} frames")

    # ── skeleton mode: reveal ink along the skeleton pixel path (the pen tip follows the real skeleton) ──
    def _lay_down_ink_skeleton(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """
        Skeleton-mode ink pass: the pen tip slides along the skeleton pixels and _reveal_ink_segment reveals the source-image ink.
        No block reveal (_ink_stamp) is done, because the skeleton is already pixel-accurate and cell completeness need not be guaranteed.
        Pen lifts (pen_lifts) are marked between strokes, and interpolation is skipped there.
        """
        strokes = self.skeleton_strokes
        if not strokes:
            return self._lay_down_ink_grid(writer, target_frames)

        # flatten the strokes into one continuous sample-point sequence, marking a pen lift between strokes
        samples: list[tuple[int, int]] = []
        pen_lifts: set[int] = set()
        for si, stroke in enumerate(strokes):
            if si > 0:
                pen_lifts.add(len(samples))  # pen lift between strokes
            samples.extend(stroke)

        n = len(samples)
        if n == 0:
            print("  no skeleton strokes, skipping the ink pass")
            for _ in range(target_frames):
                writer.write(self._snapshot_with_tip(self.out_w // 2, self.out_h // 2))
            return

        sample_idx_for_frame = self._frame_progress_indices(n, target_frames)

        # adaptive pauses (density is judged by stroke count rather than cell count)
        pause_frames = self._pause_frame_indices(target_frames, len(strokes))
        if pause_frames:
            print(f"  adaptive pauses: {len(pause_frames)} frames frozen (mode={self.cfg.pause_mode})")

        written = 0
        last_sample_idx: int | None = None
        report_step = max(1, target_frames // 10)
        for fi, si in enumerate(sample_idx_for_frame):
            # pause frame: pen tip frozen
            if fi in pause_frames and last_sample_idx is not None:
                sx, sy = samples[last_sample_idx]
                writer.write(self._snapshot_with_tip(sx, sy))
                written += 1
                if (fi + 1) % report_step == 0:
                    print(f"  ink pass progress: {int((fi + 1) / target_frames * 100)}%")
                continue

            # reveal ink along the skeleton: from the previous frame's sample point to the current frame's, reveal the source-image ink segment by segment
            if last_sample_idx is None:
                self._reveal_ink_segment(samples[si], samples[si])
            else:
                for idx in range(last_sample_idx + 1, si + 1):
                    if idx in pen_lifts:
                        continue
                    self._reveal_ink_segment(samples[idx - 1], samples[idx])

            sx, sy = samples[si]
            writer.write(self._snapshot_with_tip(sx, sy))
            written += 1
            last_sample_idx = si
            if (fi + 1) % report_step == 0:
                print(f"  ink pass progress: {int((fi + 1) / target_frames * 100)}%")

        # final safety net: pad out the frame count
        last = samples[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  ink pass done (skeleton): {n} sample points, {written} frames")

    # ── Colour-pass entry point: dispatch to the matching style by color_fill ──
    def wash_color(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        if self.cfg.color_fill == "contour-wipe":
            return self.wash_color_contour(writer, target_frames)
        return self.wash_color_brush(writer, target_frames)

    # ── brush: paint the original colours along the stroke trajectory with a circular ink brush (default style) ──
    def wash_color_brush(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        path = self.stroke_path
        n = len(path)
        disk = _feathered_disk(self.cfg.brush_radius)
        if n == 0:
            print("  no ink, skipping the colour pass")
            gaze = self.color_img
            for _ in range(target_frames):
                writer.write(gaze)
            return

        centers = [self._cell_center(cell) for cell in path]
        cell_idx_for_frame = self._frame_progress_indices(n, target_frames)

        written = 0
        last_cell_idx: int | None = None
        for fi, ci in enumerate(cell_idx_for_frame):
            # colour according to the current frame's progress: top up from the previous frame's cell to the current one,
            # also painting any intermediate cells that downsampling may have skipped, so the original colours stay continuous.
            if last_cell_idx is None:
                self._color_stamp(*centers[ci], disk)
            else:
                for cell_idx in range(last_cell_idx + 1, ci + 1):
                    self._color_stamp(*centers[cell_idx], disk)

            cx, cy = centers[ci]
            writer.write(self._snapshot_with_tip(cx, cy))
            written += 1
            last_cell_idx = ci
            if (fi + 1) % max(1, target_frames // 10) == 0:
                print(f"  colour pass progress: {int((fi + 1) / target_frames * 100)}%")

        # final safety net
        last = centers[-1]
        while written < target_frames:
            writer.write(self._snapshot_with_tip(*last))
            written += 1
        print(f"  colour pass done: {n} cells, {written} frames")

    # ── contour-wipe: contour-aware top-to-bottom sweep colouring ──
    def wash_color_contour(self, writer: cv2.VideoWriter, target_frames: int) -> None:
        """
        Colour is not painted along the stroke trajectory; instead a reveal front sweeps globally from top to bottom.
        The front stalls at a contour first (resistance about 1 deducts delay_px), then slowly crosses it along the decaying shadow below,
        giving the impression of colour creeping along the lines. The pen tip sweeps back and forth horizontally, simulating a hand colouring in.
        """
        cfg = self.cfg
        h, w = self.out_h, self.out_w

        if target_frames <= 0:
            print("  no colour-pass frames, skipping the contour-wipe phase")
            return

        # one-off precomputation: resistance field, wave boundary, deducted pixel count, row coordinate grid
        resistance = self._build_resistance_field()
        wave = _build_wipe_wave(w)
        delay_px = int(np.clip(h * cfg.wipe_delay_ratio, 12, 52))
        blocks = max(1, cfg.wipe_blocks)
        ys = np.arange(h, dtype=np.float32)[:, None]   # (H,1), reused every frame

        # reset self.drawn to the 'line art fully drawn' state (brush continues after lay_down_ink; this continues the same way)
        # color_img is the reveal target
        color_src = self.color_img.astype(np.float32)

        print(f"  contour-wipe: {w}x{h}, delay_px={delay_px}, sweeps={blocks}")

        written = 0
        # the reveal front sweeps from -delay_px to h+delay_px, covering everything
        sweep = h + 2 * delay_px
        report_step = max(1, target_frames // 10)

        for fi in range(target_frames):
            # global progress (with sine easing): 0 -> 1
            if target_frames == 1:
                progress = 1.0
            else:
                progress = fi / (target_frames - 1)
            lead = _ease_in_out_sine(progress) * sweep - delay_px

            # reveal mask: y <= lead + wave[x] - resistance[y,x]*delay_px
            threshold = lead + wave[None, :] - resistance * delay_px  # (H,W)
            reveal = ys <= threshold                        # (H,W) bool

            # reveal the original colours into the drawn buffer
            self.drawn[reveal] = color_src[reveal]

            # horizontal pen-tip sweep: blocks back-and-forth passes, odd passes go in reverse
            lane = (fi / blocks * 2.0) % 1.0               # normalised progress within one pass, 0..1
            lane = _ease_in_out_sine(lane)
            forward = (int(fi // blocks) % 2 == 0)         # even passes go forward, odd passes go in reverse
            cursor_x = int(lane * w) if forward else int((1.0 - lane) * w)
            cursor_x = max(0, min(w - 1, cursor_x))

            # cursor y = the bottom-most revealed pixel row in the current column
            col_revealed = np.where(reveal[:, cursor_x])[0]
            cursor_y = int(col_revealed[-1]) if col_revealed.size > 0 else 0

            writer.write(self._snapshot_with_tip(cursor_x, cursor_y))
            written += 1
            if (fi + 1) % report_step == 0:
                print(f"  colour pass progress (contour-wipe): {int((fi + 1) / target_frames * 100)}%")

        # final safety net: make sure the whole image is revealed (at progress=1 on the last frame lead is about h+delay_px, which in theory covers everything)
        full_reveal = np.ones((h, w), dtype=bool)
        self.drawn[full_reveal] = color_src[full_reveal]
        last = self._snapshot_with_tip(w // 2, h - 1)
        while written < target_frames:
            writer.write(last)
            written += 1
        print(f"  contour-wipe done: {written} frames")

    def render_to(self, raw_path: Path, total_ms: int) -> Path:
        cfg = self.cfg
        plan = plan_phases(total_ms, cfg)
        ink_cells = len(self.stroke_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, cfg.fps, (self.out_w, self.out_h))

        print(f"  ink streams: {len(self.ink_streams)}, ink cells: {ink_cells}")
        print(
            f"  duration: {total_ms}ms -> ink pass {plan.ink_frames}f / "
            f"colour pass {plan.color_frames}f / gaze {plan.gaze_frames}f (weights {plan.ratio_label})"
        )

        started = time.time()
        self.lay_down_ink(writer, plan.ink_frames)
        self.wash_color(writer, plan.color_frames)
        # gaze: the complete original image
        gaze_img = self.color_img
        for _ in range(plan.gaze_frames):
            writer.write(gaze_img)
        writer.release()
        print(f"  render time: {time.time() - started:.1f}s")
        return raw_path


# ──────────────────────────────────────────────────────────────
# Transcoding (system ffmpeg preferred, PyAV as fallback; if neither exists, keep mp4v)
# ──────────────────────────────────────────────────────────────
def transcode_h264(src: Path, dst: Path) -> Path:
    """
    Transcode the raw mp4v video to H.264 (yuv420p) for better player compatibility.

    Priority:
      1. System ffmpeg subprocess (best encoding efficiency, smallest size, CRF=20)
      2. PyAV (pure pip install, no system ffmpeg needed; slightly less efficient, uses CRF=28 to control size)
      3. Neither available: keep the original mp4v encoding and warn
    """
    # Path 1: system ffmpeg (recommended, smallest size)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "libx264",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(dst),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            src.unlink(missing_ok=True)
            print(f"  H.264 transcode done (ffmpeg): {dst}")
            return dst
        print(f"  [warn] ffmpeg transcode failed: {res.stderr.strip()}")

    # Path 2: PyAV (fallback, pure pip install)
    try:
        return _transcode_with_pyav(src, dst)
    except ImportError:
        pass
    except Exception as e:
        print(f"  [warn] PyAV transcode failed: {e}")

    # Path 3: neither available, keep mp4v
    print(f"  [warn] ffmpeg and PyAV not found, keeping the original mp4v encoding: {src}")
    print(f"         install either to get H.264: pip install av  or install system ffmpeg")
    return src


def _transcode_with_pyav(src: Path, dst: Path) -> Path:
    """
    Transcode to H.264 inside Python with PyAV. Raises ImportError if PyAV is not installed.
    PyAV's bundled libx264 is less efficient than system ffmpeg (several times larger at the same CRF),
    so CRF=28 is used to balance size and quality.
    """
    import av
    input_container = av.open(str(src), mode="r")
    in_stream = input_container.streams.video[0]
    width = in_stream.codec_context.width
    height = in_stream.codec_context.height
    fps = in_stream.average_rate

    output_container = av.open(str(dst), mode="w")
    out_stream = output_container.add_stream("h264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"
    out_stream.options = {"crf": "28", "preset": "medium"}

    for frame in input_container.decode(video=0):
        packet = out_stream.encode(frame)
        if packet:
            output_container.mux(packet)
    # flush
    packet = out_stream.encode(None)
    if packet:
        output_container.mux(packet)

    output_container.close()
    input_container.close()
    src.unlink(missing_ok=True)
    print(f"  H.264 transcode done (PyAV): {dst}")
    return dst


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render a single image into a streaming pen-stroke whiteboard animation video"
    )
    p.add_argument("image", help="input image path (PNG/JPG/JPEG/BMP/TIFF)")
    p.add_argument("--out-dir", default="./out", help="output directory (default: ./out)")
    p.add_argument("--total-ms", type=int, default=10000, help="total video duration in milliseconds (default: 10000)")
    p.add_argument("--bare-tip", action="store_true", help="do not overlay the pen tip/hand")
    p.add_argument(
        "--pen-image", default=str(DEFAULT_HAND_PNG),
        help="custom pen-tip/hand asset path (default: the skill's built-in drawing-hand.png)",
    )
    p.add_argument("--fps", type=int, default=None, help="override the default frame rate")
    p.add_argument("--grid-edge", type=int, default=None, help="override the default grid edge length")
    p.add_argument("--brush-radius", type=int, default=None, help="override the default ink brush radius")
    p.add_argument(
        "--color-fill", default="contour-wipe", choices=["brush", "contour-wipe"],
        help="colour-pass colouring style: contour-wipe contour-aware top-to-bottom sweep (default); brush paints along the stroke trajectory",
    )
    p.add_argument(
        "--wipe-decay", type=float, default=None,
        help="contour-wipe: resistance-field per-row downward decay factor (default 0.86; smaller crosses contours faster)",
    )
    p.add_argument(
        "--wipe-delay-ratio", type=float, default=None,
        help="contour-wipe: fraction of the front deducted at contours, times h (default 0.04; larger lingers longer at contours)",
    )
    p.add_argument(
        "--wipe-blocks", type=int, default=None,
        help="contour-wipe: number of horizontal back-and-forth pen-tip sweeps (default 18)",
    )
    p.add_argument(
        "--pause", default="heavy", choices=["auto", "off", "light", "heavy"],
        help="ink-pass pause rhythm: heavy pronounced (default); auto picks a tier by density; off disables; light a few",
    )
    p.add_argument(
        "--ink-path", default="grid", choices=["grid", "skeleton"],
        help="ink-pass stroke path: grid cell-centre interpolation (default); skeleton skeleton-level pixel tracing (follows the lines more precisely)",
    )
    return p.parse_args(argv)


def _build_cfg(args: argparse.Namespace) -> Config:
    kw: dict = {}
    if args.fps is not None:
        kw["fps"] = args.fps
    if args.grid_edge is not None:
        kw["grid_edge"] = args.grid_edge
    if args.brush_radius is not None:
        kw["brush_radius"] = args.brush_radius
    if args.color_fill is not None:
        kw["color_fill"] = args.color_fill
    if args.wipe_decay is not None:
        kw["wipe_decay"] = args.wipe_decay
    if args.wipe_delay_ratio is not None:
        kw["wipe_delay_ratio"] = args.wipe_delay_ratio
    if args.wipe_blocks is not None:
        kw["wipe_blocks"] = args.wipe_blocks
    if args.pause is not None:
        kw["pause_mode"] = args.pause
    if args.ink_path is not None:
        kw["ink_path_mode"] = args.ink_path
    return Config(**kw)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _build_cfg(args)

    print("=" * 56)
    print("Streaming pen-stroke animation renderer")
    print("=" * 56)

    image_bgr = _imread_any(args.image)
    if image_bgr is None:
        print(f"[err] cannot read image: {args.image}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = out_dir / f"stream_{ts}.mp4"
    h264_path = out_dir / f"stream_{ts}_h264.mp4"

    pen_png = Path(args.pen_image) if args.pen_image else None
    renderer = StreamBoardRenderer(image_bgr, cfg, pen_png, args.bare_tip)
    print(f"  input: {args.image}")
    print(f"  output size: {renderer.out_w}x{renderer.out_h}, fps: {cfg.fps}")

    renderer.render_to(raw_path, args.total_ms)
    final = transcode_h264(raw_path, h264_path)

    size_mb = final.stat().st_size / (1024 * 1024)
    print(f"\nFinal video: {final}")
    print(f"  file size: {size_mb:.2f} MB")
    print("=" * 56)
    print("Done")
    # the last line prints the final path so the caller can capture it
    print(f"OUTPUT={final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
