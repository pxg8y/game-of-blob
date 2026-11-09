"""
High-performance O(1) Spatial Hash Grid and 16-Thread Numba JIT Kernels
for scaling Blob Life to 5,000-20,000+ Blobs and 40,000x30,000+ Worlds.
"""

import math
from typing import Tuple, Optional
import numpy as np
from numba import njit, prange


SPATIAL_CELL_SIZE = np.float32(256.0)
INV_SPATIAL_CELL = np.float32(1.0 / 256.0)


def build_obstacle_csr_grid(
    obstacles_x1: np.ndarray,
    obstacles_y1: np.ndarray,
    obstacles_dx: np.ndarray,
    obstacles_dy: np.ndarray,
    obstacles_r: np.ndarray,
    pad_extra: float,
    w: float,
    h: float,
    cell_size: float = 256.0,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """
    Build a 2D Compressed Sparse Row (CSR) spatial lookup grid for line-segment obstacles.
    Returns (offsets, indices, inv_cell, gw, gh).
    - offsets: (gw * gh + 1,) int32 array of slice boundaries
    - indices: (total_cell_entries,) int32 array of obstacle indices
    """
    n_obs = len(obstacles_x1)
    inv_c = np.float32(1.0 / cell_size)
    gw = max(1, int(w * inv_c) + 1)
    gh = max(1, int(h * inv_c) + 1)
    num_cells = gw * gh

    if n_obs == 0:
        return np.zeros(num_cells + 1, dtype=np.int32), np.empty(0, dtype=np.int32), inv_c, gw, gh

    counts = np.zeros(num_cells, dtype=np.int32)
    for i in range(n_obs):
        x1, y1 = obstacles_x1[i], obstacles_y1[i]
        x2, y2 = x1 + obstacles_dx[i], y1 + obstacles_dy[i]
        pad = obstacles_r[i] + pad_extra
        min_x = (x1 if x1 < x2 else x2) - pad
        max_x = (x2 if x1 < x2 else x1) + pad
        min_y = (y1 if y1 < y2 else y2) - pad
        max_y = (y2 if y1 < y2 else y1) + pad
        gx0 = max(0, min(gw - 1, int(min_x * inv_c)))
        gx1 = max(0, min(gw - 1, int(max_x * inv_c)))
        gy0 = max(0, min(gh - 1, int(min_y * inv_c)))
        gy1 = max(0, min(gh - 1, int(max_y * inv_c)))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                counts[gx * gh + gy] += 1

    offsets = np.zeros(num_cells + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    indices = np.empty(offsets[-1], dtype=np.int32)
    curr = offsets[:-1].copy()

    for i in range(n_obs):
        x1, y1 = obstacles_x1[i], obstacles_y1[i]
        x2, y2 = x1 + obstacles_dx[i], y1 + obstacles_dy[i]
        pad = obstacles_r[i] + pad_extra
        min_x = (x1 if x1 < x2 else x2) - pad
        max_x = (x2 if x1 < x2 else x1) + pad
        min_y = (y1 if y1 < y2 else y2) - pad
        max_y = (y2 if y1 < y2 else y1) + pad
        gx0 = max(0, min(gw - 1, int(min_x * inv_c)))
        gx1 = max(0, min(gw - 1, int(max_x * inv_c)))
        gy0 = max(0, min(gh - 1, int(min_y * inv_c)))
        gy1 = max(0, min(gh - 1, int(max_y * inv_c)))
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                c = gx * gh + gy
                indices[curr[c]] = i
                curr[c] += 1

    return offsets, indices, inv_c, gw, gh


@njit(fastmath=True, cache=True)
def build_spatial_linked_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    inv_cell: float,
    gw: int,
    gh: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an O(N) cell-linked list spatial index over 2D coordinates `(xs, ys)`.
    Returns `(cell_head, next_idx)` where `cell_head[gx * gh + gy]` is the first entity
    index in cell `(gx, gy)` and `next_idx[i]` points to the next entity in the same cell (-1 at end).
    """
    n = len(xs)
    num_cells = gw * gh
    cell_head = np.full(num_cells, -1, dtype=np.int32)
    next_idx = np.empty(n, dtype=np.int32)
    max_gx = gw - 1
    max_gy = gh - 1

    for i in range(n):
        gx = int(xs[i] * inv_cell)
        gy = int(ys[i] * inv_cell)
        if gx < 0:
            gx = 0
        elif gx > max_gx:
            gx = max_gx
        if gy < 0:
            gy = 0
        elif gy > max_gy:
            gy = max_gy
        c = gx * gh + gy
        next_idx[i] = cell_head[c]
        cell_head[c] = i

    return cell_head, next_idx


@njit(parallel=True, fastmath=True, cache=True)
def query_nearest_foods_grid(
    blob_xs: np.ndarray,
    blob_ys: np.ndarray,
    eff_visions: np.ndarray,
    food_xs: np.ndarray,
    food_ys: np.ndarray,
    f_head: np.ndarray,
    f_next: np.ndarray,
    inv_cell: float,
    gw: int,
    gh: int,
    rock_x1: np.ndarray,
    rock_y1: np.ndarray,
    rock_dx: np.ndarray,
    rock_dy: np.ndarray,
    rock_inv_l2: np.ndarray,
    rock_r: np.ndarray,
    rock_offsets: Optional[np.ndarray] = None,
    rock_indices: Optional[np.ndarray] = None,
    rock_inv_cell: float = 0.0,
    rock_gw: int = 0,
    rock_gh: int = 0,
) -> np.ndarray:
    """
    Parallel O(1) spatial grid query finding the nearest visible food item for each blob,
    including midpoint line-of-sight rock occlusion.
    """
    n_blobs = len(blob_xs)
    food_vecs = np.zeros((n_blobs, 3), dtype=np.float32)
    if len(food_xs) == 0 or n_blobs == 0:
        return food_vecs

    n_rocks = len(rock_x1)
    has_rock_grid = rock_offsets is not None and len(rock_offsets) > 0 and len(rock_indices) > 0
    max_gx = gw - 1
    max_gy = gh - 1

    for i in prange(n_blobs):
        bx = blob_xs[i]
        by = blob_ys[i]
        vis = eff_visions[i]
        vis_sq = vis * vis

        gx0 = int((bx - vis) * inv_cell)
        gx1 = int((bx + vis) * inv_cell)
        gy0 = int((by - vis) * inv_cell)
        gy1 = int((by + vis) * inv_cell)
        if gx0 < 0:
            gx0 = 0
        if gx1 > max_gx:
            gx1 = max_gx
        if gy0 < 0:
            gy0 = 0
        if gy1 > max_gy:
            gy1 = max_gy

        best_d2 = vis_sq + 1.0
        best_f = -1
        best_dfx = np.float32(0.0)
        best_dfy = np.float32(0.0)

        for gx in range(gx0, gx1 + 1):
            col_off = gx * gh
            for gy in range(gy0, gy1 + 1):
                f_idx = f_head[col_off + gy]
                while f_idx != -1:
                    dfx = food_xs[f_idx] - bx
                    dfy = food_ys[f_idx] - by
                    d2 = dfx * dfx + dfy * dfy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_f = f_idx
                        best_dfx = dfx
                        best_dfy = dfy
                    f_idx = f_next[f_idx]

        if best_f != -1 and best_d2 <= vis_sq and best_d2 > 1e-8:
            dist = math.sqrt(best_d2)
            inv_d = np.float32(1.0 / dist)
            dx_u = best_dfx * inv_d
            dy_u = best_dfy * inv_d
            prox = np.float32(1.0) - np.float32(dist) / vis
            if prox < np.float32(0.0):
                prox = np.float32(0.0)

            if n_rocks > 0:
                mx = (bx + food_xs[best_f]) * np.float32(0.5)
                my = (by + food_ys[best_f]) * np.float32(0.5)
                if has_rock_grid:
                    mgx = int(mx * rock_inv_cell)
                    mgy = int(my * rock_inv_cell)
                    if 0 <= mgx < rock_gw and 0 <= mgy < rock_gh:
                        mcell = mgx * rock_gh + mgy
                        r_start = rock_offsets[mcell]
                        r_end = rock_offsets[mcell + 1]
                        for k in range(r_start, r_end):
                            r_idx = rock_indices[k]
                            rx1 = rock_x1[r_idx]
                            ry1 = rock_y1[r_idx]
                            rdx = rock_dx[r_idx]
                            rdy = rock_dy[r_idx]
                            t = ((mx - rx1) * rdx + (my - ry1) * rdy) * rock_inv_l2[r_idx]
                            if t < np.float32(0.0):
                                t = np.float32(0.0)
                            elif t > np.float32(1.0):
                                t = np.float32(1.0)
                            cx = rx1 + t * rdx
                            cy = ry1 + t * rdy
                            thresh = rock_r[r_idx] + np.float32(4.0)
                            if (mx - cx) ** 2 + (my - cy) ** 2 < thresh * thresh:
                                prox *= np.float32(0.25)
                                break
                else:
                    for r_idx in range(n_rocks):
                        rx1 = rock_x1[r_idx]
                        ry1 = rock_y1[r_idx]
                        rdx = rock_dx[r_idx]
                        rdy = rock_dy[r_idx]
                        t = ((mx - rx1) * rdx + (my - ry1) * rdy) * rock_inv_l2[r_idx]
                        if t < np.float32(0.0):
                            t = np.float32(0.0)
                        elif t > np.float32(1.0):
                            t = np.float32(1.0)
                        cx = rx1 + t * rdx
                        cy = ry1 + t * rdy
                        thresh = rock_r[r_idx] + np.float32(4.0)
                        if (mx - cx) ** 2 + (my - cy) ** 2 < thresh * thresh:
                            prox *= np.float32(0.25)
                            break

            food_vecs[i, 0] = dx_u
            food_vecs[i, 1] = dy_u
            food_vecs[i, 2] = prox

    return food_vecs


@njit(parallel=True, fastmath=True, cache=True)
def query_nearest_peers_grid(
    blob_xs: np.ndarray,
    blob_ys: np.ndarray,
    eff_visions: np.ndarray,
    total_foods: np.ndarray,
    moneys: np.ndarray,
    group_ids: np.ndarray,
    b_head: np.ndarray,
    b_next: np.ndarray,
    inv_cell: float,
    gw: int,
    gh: int,
    half_price: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel O(1) spatial grid query finding the nearest peer blob within vision radius.
    Returns `(peer_vecs (N, 6), nearest_peer_idxs (N,), nearest_peer_dists (N,))`.
    """
    n_blobs = len(blob_xs)
    peer_vecs = np.zeros((n_blobs, 6), dtype=np.float32)
    nearest_idxs = np.full(n_blobs, -1, dtype=np.int32)
    nearest_dists = np.full(n_blobs, 999999.0, dtype=np.float32)
    if n_blobs <= 1:
        return peer_vecs, nearest_idxs, nearest_dists

    max_gx = gw - 1
    max_gy = gh - 1
    hp_f32 = np.float32(half_price)

    for i in prange(n_blobs):
        bx = blob_xs[i]
        by = blob_ys[i]
        vis = eff_visions[i]
        vis_sq = vis * vis

        gx0 = int((bx - vis) * inv_cell)
        gx1 = int((bx + vis) * inv_cell)
        gy0 = int((by - vis) * inv_cell)
        gy1 = int((by + vis) * inv_cell)
        if gx0 < 0:
            gx0 = 0
        if gx1 > max_gx:
            gx1 = max_gx
        if gy0 < 0:
            gy0 = 0
        if gy1 > max_gy:
            gy1 = max_gy

        best_d2 = vis_sq + 1.0
        best_p = -1
        best_dpx = np.float32(0.0)
        best_dpy = np.float32(0.0)

        for gx in range(gx0, gx1 + 1):
            col_off = gx * gh
            for gy in range(gy0, gy1 + 1):
                p_idx = b_head[col_off + gy]
                while p_idx != -1:
                    if p_idx != i:
                        dpx = blob_xs[p_idx] - bx
                        dpy = blob_ys[p_idx] - by
                        d2 = dpx * dpx + dpy * dpy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_p = p_idx
                            best_dpx = dpx
                            best_dpy = dpy
                    p_idx = b_next[p_idx]

        if best_p != -1 and best_d2 <= vis_sq:
            dist = math.sqrt(best_d2)
            nearest_idxs[i] = best_p
            nearest_dists[i] = dist
            if dist > 1e-4:
                inv_d = np.float32(1.0 / dist)
                peer_vecs[i, 0] = best_dpx * inv_d
                peer_vecs[i, 1] = best_dpy * inv_d
            peer_vecs[i, 2] = max(np.float32(0.0), np.float32(1.0) - np.float32(dist) / vis)
            peer_vecs[i, 3] = np.float32(1.0) if total_foods[best_p] >= np.float32(0.5) else np.float32(-1.0)
            peer_vecs[i, 4] = np.float32(1.0) if moneys[best_p] >= hp_f32 else np.float32(-1.0)
            gi = group_ids[i]
            gp = group_ids[best_p]
            peer_vecs[i, 5] = np.float32(1.0) if (gi > 0 and gi == gp) else np.float32(-1.0)

    return peer_vecs, nearest_idxs, nearest_dists


@njit(parallel=True, fastmath=True, cache=True)
def compute_oasis_and_feelers_numba(
    blob_xs: np.ndarray,
    blob_ys: np.ndarray,
    headings: np.ndarray,
    ox: np.ndarray,
    oy: np.ndarray,
    orad: np.ndarray,
    ofert: np.ndarray,
    oring_bonus: np.ndarray,
    rock_x1: np.ndarray,
    rock_y1: np.ndarray,
    rock_dx: np.ndarray,
    rock_dy: np.ndarray,
    rock_inv_l2: np.ndarray,
    rock_r: np.ndarray,
    rock_feeler_r2: np.ndarray,
    swamp_x1: np.ndarray,
    swamp_y1: np.ndarray,
    swamp_dx: np.ndarray,
    swamp_dy: np.ndarray,
    swamp_inv_l2: np.ndarray,
    swamp_r: np.ndarray,
    swamp_feeler_r2: np.ndarray,
    rock_offsets: Optional[np.ndarray] = None,
    rock_indices: Optional[np.ndarray] = None,
    rock_inv_cell: float = 0.0,
    rock_gw: int = 0,
    rock_gh: int = 0,
    swamp_offsets: Optional[np.ndarray] = None,
    swamp_indices: Optional[np.ndarray] = None,
    swamp_inv_cell: float = 0.0,
    swamp_gw: int = 0,
    swamp_gh: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel 16-thread computation of oasis scent vectors and 3-angle feeler raycasts
    without allocating intermediate (N_blobs, N_obstacles) matrices.
    """
    n_blobs = len(blob_xs)
    oasis_vecs = np.zeros((n_blobs, 3), dtype=np.float32)
    feeler_vecs = np.zeros((n_blobs, 3), dtype=np.float32)
    if n_blobs == 0:
        return oasis_vecs, feeler_vecs

    n_oases = len(ox)
    n_rocks = len(rock_x1)
    n_swamps = len(swamp_x1)
    has_rock_grid = rock_offsets is not None and len(rock_offsets) > 0 and len(rock_indices) > 0
    has_swamp_grid = swamp_offsets is not None and len(swamp_offsets) > 0 and len(swamp_indices) > 0
    angles = (-0.65, 0.0, 0.65)

    oasis_weights = np.empty(n_oases, dtype=np.float32)
    for o_idx in range(n_oases):
        oasis_weights[o_idx] = (ofert[o_idx] ** np.float32(1.75)) * oring_bonus[o_idx]

    for i in prange(n_blobs):
        bx = blob_xs[i]
        by = blob_ys[i]
        local_fert = np.float32(-1.0)

        if n_oases > 0:
            best_score = np.float32(-1.0)
            best_dox = np.float32(0.0)
            best_doy = np.float32(0.0)
            best_dist = np.float32(1.0)

            for o_idx in range(n_oases):
                dox = ox[o_idx] - bx
                doy = oy[o_idx] - by
                odist = math.sqrt(dox * dox + doy * doy)
                f_val = ofert[o_idx]
                if odist <= orad[o_idx] * np.float32(1.25):
                    lf = f_val * np.float32(2.0) - np.float32(1.0)
                    if lf > local_fert:
                        local_fert = lf

                score = oasis_weights[o_idx] / (odist + np.float32(220.0))
                if score > best_score:
                    best_score = score
                    best_dox = dox
                    best_doy = doy
                    best_dist = max(np.float32(1e-4), np.float32(odist))

            oasis_vecs[i, 0] = best_dox / best_dist
            oasis_vecs[i, 1] = best_doy / best_dist

        oasis_vecs[i, 2] = local_fert

        if n_rocks > 0 or n_swamps > 0:
            hd = headings[i]
            for f_col in range(3):
                ang = hd + np.float32(angles[f_col])
                px = bx + np.float32(math.cos(ang) * 72.0)
                py = by + np.float32(math.sin(ang) * 72.0)

                min_rock_d = np.float32(1e9)
                hit_rock = False
                if n_rocks > 0:
                    if has_rock_grid:
                        pgx = int(px * rock_inv_cell)
                        pgy = int(py * rock_inv_cell)
                        if 0 <= pgx < rock_gw and 0 <= pgy < rock_gh:
                            pcell = pgx * rock_gh + pgy
                            r_start = rock_offsets[pcell]
                            r_end = rock_offsets[pcell + 1]
                            for k in range(r_start, r_end):
                                r_idx = rock_indices[k]
                                rx1 = rock_x1[r_idx]
                                ry1 = rock_y1[r_idx]
                                rdx = rock_dx[r_idx]
                                rdy = rock_dy[r_idx]
                                t = ((px - rx1) * rdx + (py - ry1) * rdy) * rock_inv_l2[r_idx]
                                if t < np.float32(0.0):
                                    t = np.float32(0.0)
                                elif t > np.float32(1.0):
                                    t = np.float32(1.0)
                                cx = rx1 + t * rdx
                                cy = ry1 + t * rdy
                                d2 = (px - cx) ** 2 + (py - cy) ** 2
                                if d2 < rock_feeler_r2[r_idx]:
                                    hit_rock = True
                                    dr = np.float32(math.sqrt(d2)) - rock_r[r_idx]
                                    if dr < min_rock_d:
                                        min_rock_d = dr
                    else:
                        for r_idx in range(n_rocks):
                            rx1 = rock_x1[r_idx]
                            ry1 = rock_y1[r_idx]
                            rdx = rock_dx[r_idx]
                            rdy = rock_dy[r_idx]
                            rx2 = rx1 + rdx
                            r_pad = np.float32(56.0) + rock_r[r_idx]
                            rx_min = (rx1 if rx1 < rx2 else rx2) - r_pad
                            rx_max = (rx2 if rx1 < rx2 else rx1) + r_pad
                            if px < rx_min or px > rx_max:
                                continue
                            ry2 = ry1 + rdy
                            ry_min = (ry1 if ry1 < ry2 else ry2) - r_pad
                            ry_max = (ry2 if ry1 < ry2 else ry1) + r_pad
                            if py < ry_min or py > ry_max:
                                continue
                            t = ((px - rx1) * rdx + (py - ry1) * rdy) * rock_inv_l2[r_idx]
                            if t < np.float32(0.0):
                                t = np.float32(0.0)
                            elif t > np.float32(1.0):
                                t = np.float32(1.0)
                            cx = rx1 + t * rdx
                            cy = ry1 + t * rdy
                            d2 = (px - cx) ** 2 + (py - cy) ** 2
                            if d2 < rock_feeler_r2[r_idx]:
                                hit_rock = True
                                dr = np.float32(math.sqrt(d2)) - rock_r[r_idx]
                                if dr < min_rock_d:
                                    min_rock_d = dr

                if hit_rock:
                    val = np.float32(1.0) - min_rock_d * np.float32(1.0 / 56.0)
                    if val < np.float32(0.15):
                        val = np.float32(0.15)
                    elif val > np.float32(1.0):
                        val = np.float32(1.0)
                    feeler_vecs[i, f_col] = val
                elif n_swamps > 0:
                    min_swamp_d = np.float32(1e9)
                    hit_swamp = False
                    if has_swamp_grid:
                        sgx = int(px * swamp_inv_cell)
                        sgy = int(py * swamp_inv_cell)
                        if 0 <= sgx < swamp_gw and 0 <= sgy < swamp_gh:
                            scell = sgx * swamp_gh + sgy
                            s_start = swamp_offsets[scell]
                            s_end = swamp_offsets[scell + 1]
                            for k in range(s_start, s_end):
                                s_idx = swamp_indices[k]
                                sx1 = swamp_x1[s_idx]
                                sy1 = swamp_y1[s_idx]
                                sdx = swamp_dx[s_idx]
                                sdy = swamp_dy[s_idx]
                                t = ((px - sx1) * sdx + (py - sy1) * sdy) * swamp_inv_l2[s_idx]
                                if t < np.float32(0.0):
                                    t = np.float32(0.0)
                                elif t > np.float32(1.0):
                                    t = np.float32(1.0)
                                cx = sx1 + t * sdx
                                cy = sy1 + t * sdy
                                d2 = (px - cx) ** 2 + (py - cy) ** 2
                                if d2 < swamp_feeler_r2[s_idx]:
                                    hit_swamp = True
                                    ds = np.float32(math.sqrt(d2)) - swamp_r[s_idx]
                                    if ds < min_swamp_d:
                                        min_swamp_d = ds
                    else:
                        for s_idx in range(n_swamps):
                            sx1 = swamp_x1[s_idx]
                            sy1 = swamp_y1[s_idx]
                            sdx = swamp_dx[s_idx]
                            sdy = swamp_dy[s_idx]
                            sx2 = sx1 + sdx
                            s_pad = np.float32(48.0) + swamp_r[s_idx]
                            sx_min = (sx1 if sx1 < sx2 else sx2) - s_pad
                            sx_max = (sx2 if sx1 < sx2 else sx1) + s_pad
                            if px < sx_min or px > sx_max:
                                continue
                            sy2 = sy1 + sdy
                            sy_min = (sy1 if sy1 < sy2 else sy2) - s_pad
                            sy_max = (sy2 if sy1 < sy2 else sy1) + s_pad
                            if py < sy_min or py > sy_max:
                                continue
                            t = ((px - sx1) * sdx + (py - sy1) * sdy) * swamp_inv_l2[s_idx]
                            if t < np.float32(0.0):
                                t = np.float32(0.0)
                            elif t > np.float32(1.0):
                                t = np.float32(1.0)
                            cx = sx1 + t * sdx
                            cy = sy1 + t * sdy
                            d2 = (px - cx) ** 2 + (py - cy) ** 2
                            if d2 < swamp_feeler_r2[s_idx]:
                                hit_swamp = True
                                ds = np.float32(math.sqrt(d2)) - swamp_r[s_idx]
                                if ds < min_swamp_d:
                                    min_swamp_d = ds

                    if hit_swamp:
                        val_s = np.float32(1.0) - min_swamp_d * np.float32(1.0 / 48.0)
                        if val_s < np.float32(0.15):
                            val_s = np.float32(0.15)
                        elif val_s > np.float32(0.85):
                            val_s = np.float32(0.85)
                        feeler_vecs[i, f_col] = -val_s

    return oasis_vecs, feeler_vecs


@njit(parallel=True, fastmath=True, cache=True)
def find_foraging_candidates_grid(
    food_xs: np.ndarray,
    food_ys: np.ndarray,
    extra_reaches: np.ndarray,
    blob_xs: np.ndarray,
    blob_ys: np.ndarray,
    blob_reaches: np.ndarray,
    b_head: np.ndarray,
    b_next: np.ndarray,
    inv_cell: float,
    gw: int,
    gh: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each food item, find the top-2 closest blobs within foraging reach using the O(1) blob grid.
    Returns `(cand1, cand2)` of shape `(n_foods,)` with blob indices (-1 if none).
    """
    n_foods = len(food_xs)
    cand1 = np.full(n_foods, -1, dtype=np.int32)
    cand2 = np.full(n_foods, -1, dtype=np.int32)
    if n_foods == 0 or len(blob_xs) == 0:
        return cand1, cand2

    max_gx = gw - 1
    max_gy = gh - 1
    max_possible_reach = np.float32(68.0)

    for f_idx in prange(n_foods):
        fx = food_xs[f_idx]
        fy = food_ys[f_idx]
        ex_r = extra_reaches[f_idx]

        gx0 = int((fx - max_possible_reach) * inv_cell)
        gx1 = int((fx + max_possible_reach) * inv_cell)
        gy0 = int((fy - max_possible_reach) * inv_cell)
        gy1 = int((fy + max_possible_reach) * inv_cell)
        if gx0 < 0:
            gx0 = 0
        if gx1 > max_gx:
            gx1 = max_gx
        if gy0 < 0:
            gy0 = 0
        if gy1 > max_gy:
            gy1 = max_gy

        best1_d2 = np.float32(1e12)
        best2_d2 = np.float32(1e12)
        best1_b = -1
        best2_b = -1

        for gx in range(gx0, gx1 + 1):
            col_off = gx * gh
            for gy in range(gy0, gy1 + 1):
                b_idx = b_head[col_off + gy]
                while b_idx != -1:
                    dx = fx - blob_xs[b_idx]
                    dy = fy - blob_ys[b_idx]
                    d2 = dx * dx + dy * dy
                    reach = ex_r + blob_reaches[b_idx]
                    if d2 <= reach * reach:
                        if d2 < best1_d2:
                            best2_d2 = best1_d2
                            best2_b = best1_b
                            best1_d2 = d2
                            best1_b = b_idx
                        elif d2 < best2_d2:
                            best2_d2 = d2
                            best2_b = b_idx
                    b_idx = b_next[b_idx]

        cand1[f_idx] = best1_b
        cand2[f_idx] = best2_b

    return cand1, cand2


@njit(parallel=True, fastmath=True, cache=True)
def parallel_mlp_layer_forward(
    a_in: np.ndarray,
    w_batch: np.ndarray,
    b_batch: np.ndarray,
    out: np.ndarray,
    act_code: int,
    is_output_layer: bool,
    noise_arr: np.ndarray,
    has_noise: bool,
) -> None:
    """
    Parallel 16-thread batched GEMV + bias + activation for individual per-blob weight matrices.
    act_code: 0=tanh, 1=relu, 2=gelu, 3=silu
    """
    n, fan_in, fan_out = w_batch.shape
    sqrt_2_pi = np.float32(0.7978845608028654)
    gelu_coef = np.float32(0.044715)

    for i in prange(n):
        # 1. Initialize with biases (contiguous along j)
        for j in range(fan_out):
            out[i, j] = b_batch[i, j]

        # 2. Vectorized contiguous accumulation (k outer, j inner)
        for k in range(fan_in):
            aik = a_in[i, k]
            for j in range(fan_out):
                out[i, j] += aik * w_batch[i, k, j]

        # 3. In-place activation pass
        if is_output_layer:
            if has_noise:
                for j in range(fan_out):
                    out[i, j] = math.tanh(out[i, j] + noise_arr[i, j])
            else:
                for j in range(fan_out):
                    out[i, j] = math.tanh(out[i, j])
        else:
            if act_code == 1:  # relu
                for j in range(fan_out):
                    val = out[i, j]
                    out[i, j] = val if val > np.float32(0.0) else np.float32(0.0)
            elif act_code == 2:  # gelu
                for j in range(fan_out):
                    acc = out[i, j]
                    x3 = acc * acc * acc
                    out[i, j] = np.float32(0.5) * acc * (
                        np.float32(1.0) + np.float32(math.tanh(sqrt_2_pi * (acc + gelu_coef * x3)))
                    )
            elif act_code == 3:  # silu
                for j in range(fan_out):
                    acc = out[i, j]
                    cl = acc
                    if cl < np.float32(-15.0):
                        cl = np.float32(-15.0)
                    elif cl > np.float32(15.0):
                        cl = np.float32(15.0)
                    out[i, j] = acc / (np.float32(1.0) + np.float32(math.exp(-cl)))
            else:  # tanh
                for j in range(fan_out):
                    out[i, j] = math.tanh(out[i, j])


@njit(parallel=True, fastmath=True, cache=True)
def parallel_mlp_backprop_delta(
    w_batch: np.ndarray,
    delta_in: np.ndarray,
    act_prev: np.ndarray,
    delta_out: np.ndarray,
    act_code: int,
) -> None:
    """Parallel 16-thread backprop sensitivity step `delta_out = (W @ delta_in) * deriv(act_prev)`."""
    n, fan_in, fan_out = w_batch.shape
    for i in prange(n):
        if fan_out == 6:
            # Output-layer fast path: preload 6 delta scalars into registers
            d0 = delta_in[i, 0]
            d1 = delta_in[i, 1]
            d2 = delta_in[i, 2]
            d3 = delta_in[i, 3]
            d4 = delta_in[i, 4]
            d5 = delta_in[i, 5]
            for k in range(fan_in):
                acc = (
                    w_batch[i, k, 0] * d0
                    + w_batch[i, k, 1] * d1
                    + w_batch[i, k, 2] * d2
                    + w_batch[i, k, 3] * d3
                    + w_batch[i, k, 4] * d4
                    + w_batch[i, k, 5] * d5
                )
                a = act_prev[i, k]
                if act_code == 1:
                    deriv = np.float32(1.0) if a > np.float32(0.0) else np.float32(0.05)
                elif act_code == 2 or act_code == 3:
                    a2 = a * a
                    deriv = np.float32(1.0) - np.float32(0.5) * (a2 / (np.float32(1.0) + a2))
                else:
                    d_val = np.float32(1.0) - a * a
                    deriv = d_val if d_val > np.float32(0.08) else np.float32(0.08)
                delta_out[i, k] = acc * deriv
        else:
            for k in range(fan_in):
                acc = np.float32(0.0)
                for j in range(fan_out):
                    acc += w_batch[i, k, j] * delta_in[i, j]

                a = act_prev[i, k]
                if act_code == 1:
                    deriv = np.float32(1.0) if a > np.float32(0.0) else np.float32(0.05)
                elif act_code == 2 or act_code == 3:
                    a2 = a * a
                    deriv = np.float32(1.0) - np.float32(0.5) * (a2 / (np.float32(1.0) + a2))
                else:
                    d_val = np.float32(1.0) - a * a
                    deriv = d_val if d_val > np.float32(0.08) else np.float32(0.08)
                delta_out[i, k] = acc * deriv


@njit(parallel=True, fastmath=True, cache=True)
def prep_step_observations_numba(
    obs_batch: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    home_xs: np.ndarray,
    home_ys: np.ndarray,
    home_radii: np.ndarray,
    foods_today: np.ndarray,
    energies: np.ndarray,
    gene_sizes: np.ndarray,
    moneys: np.ndarray,
    reputations: np.ndarray,
    peer_reputations: np.ndarray,
    home_gain: float,
    inv_900: float,
    inv_repro: float,
    is_at_home_out: np.ndarray,
    mem_xs: np.ndarray,
    mem_ys: np.ndarray,
    mem_strengths: np.ndarray,
) -> None:
    n = len(xs)
    for i in prange(n):
        dhx = home_xs[i] - xs[i]
        dhy = home_ys[i] - ys[i]
        d2 = dhx * dhx + dhy * dhy
        hr = home_radii[i]
        at_home = d2 <= (hr * hr)
        is_at_home_out[i] = 1 if at_home else 0

        if d2 > 1e-8:
            dist = math.sqrt(d2)
            inv_dh = 1.0 / dist
            raw_dx = dhx * inv_dh
            raw_dy = dhy * inv_dh
        else:
            dist = 0.0
            raw_dx = 0.0
            raw_dy = 0.0

        if at_home:
            obs_batch[i, 3] = np.float32(0.0)
            obs_batch[i, 4] = np.float32(0.0)
        else:
            obs_batch[i, 3] = np.float32(raw_dx * home_gain)
            obs_batch[i, 4] = np.float32(raw_dy * home_gain)

        dh_norm = dist * inv_900
        obs_batch[i, 5] = np.float32(0.0 if dh_norm >= 1.0 else (1.0 - dh_norm))

        sat = foods_today[i] * inv_repro
        obs_batch[i, 12] = np.float32(1.5 if sat > 1.5 else sat)
        obs_batch[i, 13] = np.float32((energies[i] / (85.0 + 15.0 * gene_sizes[i])) * 2.0 - 1.0)
        obs_batch[i, 14] = np.float32(math.tanh(moneys[i] * 0.05))

        # Spatial Memory (Channels 19, 20)
        ms = mem_strengths[i]
        if ms > 0.02:
            dmx = mem_xs[i] - xs[i]
            dmy = mem_ys[i] - ys[i]
            dist_mem = math.hypot(dmx, dmy)
            if dist_mem > 1e-4:
                inv_dm = ms / dist_mem
                obs_batch[i, 19] = np.float32(dmx * inv_dm)
                obs_batch[i, 20] = np.float32(dmy * inv_dm)
            else:
                obs_batch[i, 19] = np.float32(0.0)
                obs_batch[i, 20] = np.float32(0.0)
        else:
            obs_batch[i, 19] = np.float32(0.0)
            obs_batch[i, 20] = np.float32(0.0)

        obs_batch[i, 24] = peer_reputations[i]
        obs_batch[i, 25] = reputations[i]


@njit(parallel=True, fastmath=True, cache=True)
def apply_kinematics_and_metabolism_batch_numba(
    xs: np.ndarray,
    ys: np.ndarray,
    vxs: np.ndarray,
    vys: np.ndarray,
    headings: np.ndarray,
    energies: np.ndarray,
    in_swamps: np.ndarray,
    is_at_homes: np.ndarray,
    foods_today: np.ndarray,
    max_speeds: np.ndarray,
    body_radii: np.ndarray,
    basal_costs: np.ndarray,
    has_groups: np.ndarray,
    home_xs: np.ndarray,
    home_ys: np.ndarray,
    home_radii: np.ndarray,
    actions_batch: np.ndarray,
    rock_x1: np.ndarray,
    rock_y1: np.ndarray,
    rock_dx: np.ndarray,
    rock_dy: np.ndarray,
    rock_inv_l2: np.ndarray,
    rock_r: np.ndarray,
    swamp_x1: np.ndarray,
    swamp_y1: np.ndarray,
    swamp_dx: np.ndarray,
    swamp_dy: np.ndarray,
    swamp_inv_l2: np.ndarray,
    swamp_r: np.ndarray,
    width: float,
    height: float,
    repro_need: float,
    surv_need: float,
    ticks_per_day: float,
    season_mult: float,
    dt_scale: float,
    night_mode: bool,
    hit_rocks_out: np.ndarray,
    speed_norms_out: np.ndarray,
) -> None:
    """
    Parallel 16-thread kinematics, terrain physics, boundary reflection,
    and metabolic energy deduction for living blobs.
    """
    n = len(xs)
    blend = 0.28 if dt_scale <= 1.0 else (1.0 - (0.72 ** min(dt_scale, 10.0)))
    day_len_scale = 360.0 / max(180.0, ticks_per_day)
    n_rocks = len(rock_x1)
    n_swamps = len(swamp_x1)

    for i in prange(n):
        mx = actions_batch[i, 0]
        my = actions_batch[i, 1]
        mag = math.hypot(mx, my)
        if mag > 1.0:
            mx /= mag
            my /= mag

        sp = (actions_batch[i, 2] + 1.0) * 0.5
        sp_mult = 1.0 + 0.55 * sp
        if in_swamps[i]:
            sp_mult *= 0.44
        if is_at_homes[i] and (
            foods_today[i] >= repro_need or (night_mode and foods_today[i] >= surv_need)
        ):
            sp_mult *= 0.18

        vx = (1.0 - blend) * vxs[i] + blend * (mx * max_speeds[i] * sp_mult)
        vy = (1.0 - blend) * vys[i] + blend * (my * max_speeds[i] * sp_mult)

        hx = home_xs[i]
        hy = home_ys[i]
        hr = home_radii[i]
        step_dx = vx * dt_scale
        step_dy = vy * dt_scale

        if dt_scale > 1.0 and (night_mode or foods_today[i] >= surv_need):
            to_hx = hx - xs[i]
            to_hy = hy - ys[i]
            step_len_sq = step_dx * step_dx + step_dy * step_dy
            if step_len_sq > 1e-6 and (step_dx * to_hx + step_dy * to_hy) > 0.0:
                t_proj = min(1.0, max(0.0, (to_hx * step_dx + to_hy * step_dy) / step_len_sq))
                cx = xs[i] + t_proj * step_dx - hx
                cy = ys[i] + t_proj * step_dy - hy
                if (cx * cx + cy * cy) <= (hr * hr):
                    vx *= 0.25
                    vy *= 0.25
                    step_dx *= t_proj
                    step_dy *= t_proj

        bx = xs[i] + step_dx
        by = ys[i] + step_dy
        br = body_radii[i]

        # 1. Swamp immersion check at new coordinates
        cur_in_swamp = False
        for s in range(n_swamps):
            sx1 = swamp_x1[s]
            sy1 = swamp_y1[s]
            sdx = swamp_dx[s]
            sdy = swamp_dy[s]
            thresh = swamp_r[s] + br
            if sdx >= 0.0:
                if bx < sx1 - thresh or bx > sx1 + sdx + thresh:
                    continue
            else:
                if bx < sx1 + sdx - thresh or bx > sx1 + thresh:
                    continue
            if sdy >= 0.0:
                if by < sy1 - thresh or by > sy1 + sdy + thresh:
                    continue
            else:
                if by < sy1 + sdy - thresh or by > sy1 + thresh:
                    continue
            t = ((bx - sx1) * sdx + (by - sy1) * sdy) * swamp_inv_l2[s]
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            dx = bx - (sx1 + t * sdx)
            dy = by - (sy1 + t * sdy)
            if dx * dx + dy * dy <= thresh * thresh:
                cur_in_swamp = True
                break
        in_swamps[i] = cur_in_swamp

        # 2. Rock wall collisions & tangential sliding
        hit_rock = False
        pad_r = br + 2.0
        for r in range(n_rocks):
            rx1 = rock_x1[r]
            ry1 = rock_y1[r]
            rdx = rock_dx[r]
            rdy = rock_dy[r]
            target_r = rock_r[r] + pad_r
            if rdx >= 0.0:
                if bx < rx1 - target_r or bx > rx1 + rdx + target_r:
                    continue
            else:
                if bx < rx1 + rdx - target_r or bx > rx1 + target_r:
                    continue
            if rdy >= 0.0:
                if by < ry1 - target_r or by > ry1 + rdy + target_r:
                    continue
            else:
                if by < ry1 + rdy - target_r or by > ry1 + target_r:
                    continue
            t = ((bx - rx1) * rdx + (by - ry1) * rdy) * rock_inv_l2[r]
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            cx = rx1 + t * rdx
            cy = ry1 + t * rdy
            dx = bx - cx
            dy = by - cy
            d2 = dx * dx + dy * dy
            if d2 < target_r * target_r:
                hit_rock = True
                dist = math.sqrt(d2)
                if dist > 1e-4:
                    inv_d = 1.0 / dist
                    nx = dx * inv_d
                    ny = dy * inv_d
                else:
                    nx, ny = 1.0, 0.0
                bx = cx + nx * target_r
                by = cy + ny * target_r
                vn = vx * nx + vy * ny
                if vn < 0.0:
                    vx -= 1.15 * vn * nx
                    vy -= 1.15 * vn * ny
        hit_rocks_out[i] = 1 if hit_rock else 0

        # 3. Heading update on significant velocity
        sp_mag = math.hypot(vx, vy)
        if sp_mag > 0.08:
            headings[i] = math.atan2(vy, vx)

        # 4. Arena boundary reflection
        pad = br + 4.0
        max_x = width - pad
        if bx < pad:
            bx = pad
            vx *= -0.6
        elif bx > max_x:
            bx = max_x
            vx *= -0.6

        max_y = height - pad
        if by < pad:
            by = pad
            vy *= -0.6
        elif by > max_y:
            by = max_y
            vy *= -0.6

        xs[i] = bx
        ys[i] = by
        vxs[i] = vx
        vys[i] = vy

        # 5. Home proximity
        dhx = hx - bx
        dhy = hy - by
        at_home = (dhx * dhx + dhy * dhy) <= (hr * hr)
        is_at_homes[i] = at_home

        # 6. Metabolism energy deduction
        move_speed_norm = sp_mag / max(0.1, max_speeds[i])
        speed_norms_out[i] = move_speed_norm
        movement_cost = basal_costs[i] * (
            0.35 + 0.60 * (move_speed_norm * move_speed_norm) + 0.42 * (sp * sp)
        )
        if cur_in_swamp:
            movement_cost *= 1.75
        if at_home:
            movement_cost *= 0.25 if has_groups[i] else 0.35
        else:
            movement_cost *= season_mult
            if night_mode:
                movement_cost *= 1.18

        e_new = energies[i] - movement_cost * dt_scale * day_len_scale
        energies[i] = e_new if e_new > 0.0 else 0.0


update_blobs_kinematics_and_metabolism_kernel = apply_kinematics_and_metabolism_batch_numba


@njit(fastmath=True, cache=True)
def batch_point_hits_rock_csr(
    px: np.ndarray,
    py: np.ndarray,
    pad: float,
    rock_x1: np.ndarray,
    rock_y1: np.ndarray,
    rock_dx: np.ndarray,
    rock_dy: np.ndarray,
    rock_inv_len_sq: np.ndarray,
    rock_r: np.ndarray,
    rock_offsets: np.ndarray,
    rock_indices: np.ndarray,
    inv_cell: float,
    gw: int,
    gh: int,
) -> np.ndarray:
    n = len(px)
    mask = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        x = px[i]
        y = py[i]
        gx = int(x * inv_cell)
        gy = int(y * inv_cell)
        if gx < 0 or gx >= gw or gy < 0 or gy >= gh:
            continue
        cell_idx = gx * gh + gy
        start = rock_offsets[cell_idx]
        end = rock_offsets[cell_idx + 1]
        for k in range(start, end):
            r_idx = rock_indices[k]
            rx1 = rock_x1[r_idx]
            ry1 = rock_y1[r_idx]
            rdx = rock_dx[r_idx]
            rdy = rock_dy[r_idx]
            inv_l2 = rock_inv_len_sq[r_idx]
            rr = rock_r[r_idx] + pad
            t = ((x - rx1) * rdx + (y - ry1) * rdy) * inv_l2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            dx = x - (rx1 + t * rdx)
            dy = y - (ry1 + t * rdy)
            if dx * dx + dy * dy < rr * rr:
                mask[i] = True
                break
    return mask


@njit(fastmath=True, cache=True)
def filter_active_peers_numba(
    n: int,
    min_p_idxs: np.ndarray,
    peer_dists: np.ndarray,
    peer_reach: float,
    social_desires: np.ndarray,
    trade_desires: np.ndarray,
    foods_today: np.ndarray,
    surplus_foods: np.ndarray,
    moneys: np.ndarray,
    energies: np.ndarray,
    reputations: np.ndarray,
    group_ids: np.ndarray,
    home_ids: np.ndarray,
    trade_cooldowns: np.ndarray,
    surv_target: float,
    repro_target: float,
    raid_food_price_threshold: float,
):
    """
    Numba-accelerated active peer interaction filter.
    Filters 2,000+ blobs down to active candidates (<30 pairs/tick) in ~0.08 ms,
    completely preventing evening home congregation spikes.
    Returns (active_i, active_j, active_kind):
      bit 0 (1): social encounter
      bit 1 (2): peer trade
      bit 2 (4): peer raid
    """
    active_i = np.empty(n, dtype=np.int32)
    active_j = np.empty(n, dtype=np.int32)
    active_kind = np.empty(n, dtype=np.int32)
    count = 0

    for i in range(n):
        if peer_dists[i] >= peer_reach:
            continue
        j = min_p_idxs[i]
        if j < 0 or j >= n or i == j:
            continue
        # Natural mutual pair deduplication: only process if i < j or j's peer is not i
        if i > j and min_p_idxs[j] == i:
            continue

        gid_i = group_ids[i]
        gid_j = group_ids[j]

        # 1. Social encounter: at least one ungrouped, desire thresholds met
        can_soc = False
        if (gid_i < 0 or gid_j < 0):
            if social_desires[i] > 0.05 and social_desires[j] > -0.05:
                can_soc = True

        # 2. Peer trade
        can_tr = False
        if trade_cooldowns[i] == 0 and trade_cooldowns[j] == 0:
            rep_i = reputations[i]
            rep_j = reputations[j]
            # Boycott: outlaw (<= -0.35) and honorable/normal (> -0.05)
            is_boycott = (rep_i <= -0.35 and rep_j > -0.05) or (rep_j <= -0.35 and rep_i > -0.05)
            if not is_boycott:
                tot_i = foods_today[i] + surplus_foods[i]
                tot_j = foods_today[j] + surplus_foods[j]
                # i sells to j
                if tot_i >= 0.5 and moneys[j] >= 0.4:
                    rem_i = tot_i - 0.5
                    if (
                        ((surplus_foods[i] >= 0.5 or rem_i >= repro_target - 1e-4) and trade_desires[i] > -0.10)
                        or (rem_i >= surv_target - 1e-4 and trade_desires[i] > 0.12)
                        or (trade_desires[i] > 0.25)
                    ):
                        if foods_today[j] < repro_target or energies[j] < 75.0 or trade_desires[j] > 0.15:
                            can_tr = True
                # j sells to i
                if not can_tr and tot_j >= 0.5 and moneys[i] >= 0.4:
                    rem_j = tot_j - 0.5
                    if (
                        ((surplus_foods[j] >= 0.5 or rem_j >= repro_target - 1e-4) and trade_desires[j] > -0.10)
                        or (rem_j >= surv_target - 1e-4 and trade_desires[j] > 0.12)
                        or (trade_desires[j] > 0.25)
                    ):
                        if foods_today[i] < repro_target or energies[i] < 75.0 or trade_desires[i] > 0.15:
                            can_tr = True

        # 3. Peer raid: outsider, different home, energy >= 28, motivated
        can_rd = False
        hid_i = home_ids[i]
        hid_j = home_ids[j]
        if hid_i != hid_j and (gid_i < 0 or gid_j < 0 or gid_i != gid_j):
            if energies[i] >= 28.0:
                is_mot = (
                    (foods_today[i] < surv_target and moneys[i] < raid_food_price_threshold)
                    or (social_desires[i] < -0.16 and trade_desires[i] < -0.12 and foods_today[i] < repro_target)
                    or (reputations[j] <= -0.40 and reputations[i] >= 0.0 and social_desires[i] > 0.15 and energies[i] >= 38.0)
                )
                if is_mot:
                    can_rd = True

        if can_soc or can_tr or can_rd:
            k = 0
            if can_soc: k |= 1
            if can_tr:  k |= 2
            if can_rd:  k |= 4
            active_i[count] = i
            active_j[count] = j
            active_kind[count] = k
            count += 1

    return active_i[:count], active_j[:count], active_kind[:count]


@njit(fastmath=True, cache=True)
def filter_active_market_numba(
    n: int,
    is_at_home: np.ndarray,
    trade_cooldowns: np.ndarray,
    foods_today: np.ndarray,
    surplus_foods: np.ndarray,
    moneys: np.ndarray,
    energies: np.ndarray,
    trade_desires: np.ndarray,
    group_ids: np.ndarray,
    home_ids: np.ndarray,
    homes_gold: np.ndarray,
    homes_depot: np.ndarray,
    group_has_depot_food: np.ndarray,
    max_group_id: int,
    tod: float,
    surv_target: float,
    repro_target: float,
):
    """
    Numba-accelerated village market trade filter.
    Prunes 1,500+ blobs congregating at home down to strictly eligible buyers/sellers (<15 calls/tick).
    """
    active_indices = np.empty(n, dtype=np.int32)
    count = 0
    tod_gt_55 = tod > 0.55

    for i in range(n):
        if is_at_home[i] == 0 or trade_cooldowns[i] > 0:
            continue

        hid = home_ids[i]
        h_gold = homes_gold[hid]
        h_depot = homes_depot[hid]

        ft = foods_today[i]
        sf = surplus_foods[i]
        tot_f = ft + sf
        td = trade_desires[i]

        # Selling eligibility
        can_sell = ((sf >= 0.5 and td >= -0.15) or (tot_f >= 0.5 and td > 0.25)) and (h_gold >= 0.3)

        # Buying eligibility
        can_buy = False
        if moneys[i] >= 0.4:
            has_depot = (h_depot >= 0.5)
            if not has_depot:
                gid = group_ids[i]
                if gid >= 0 and gid <= max_group_id and group_has_depot_food[gid]:
                    has_depot = True
            if has_depot and ft < repro_target:
                needs_survival = (ft < surv_target) and (tod_gt_55 or energies[i] < 55.0)
                close_to_repro = (ft >= surv_target) and ((repro_target - ft) <= 0.5)
                wants_buy = td > 0.25
                if needs_survival or close_to_repro or wants_buy:
                    can_buy = True

        if can_sell or can_buy:
            active_indices[count] = i
            count += 1

    return active_indices[:count]


@njit(fastmath=True, cache=True)
def mutate_neural_layer_numba(
    w: np.ndarray,
    b: np.ndarray,
    partner_w: np.ndarray,
    partner_b: np.ndarray,
    has_partner: bool,
    mutation_rate: float,
    mutation_strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast JIT-compiled crossover and Gaussian mutation for neural layer arrays."""
    out_w = w.copy()
    out_b = b.copy()
    rows, cols = out_w.shape
    if has_partner:
        for j in range(cols):
            if np.random.random() < 0.35:
                for i in range(rows):
                    out_w[i, j] = partner_w[i, j]
                out_b[j] = partner_b[j]
    if mutation_rate > 0.0 and mutation_strength > 0.0:
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < mutation_rate:
                    val = out_w[i, j] + np.random.normal() * mutation_strength
                    if val < -3.5:
                        val = -3.5
                    elif val > 3.5:
                        val = 3.5
                    out_w[i, j] = val
        for j in range(cols):
            if np.random.random() < mutation_rate:
                val = out_b[j] + np.random.normal() * mutation_strength
                if val < -2.5:
                    val = -2.5
                elif val > 2.5:
                    val = 2.5
                out_b[j] = val
    return out_w, out_b


@njit(fastmath=True, nogil=True, cache=True)
def pack_binary_blobs_kernel(
    out_buf: np.ndarray,          # (N, 8) float32
    xs: np.ndarray,               # (N,) float64
    ys: np.ndarray,               # (N,) float64
    headings: np.ndarray,         # (N,) float32
    radii: np.ndarray,            # (N,) float32
    foods_today: np.ndarray,      # (N,) float32
    ids: np.ndarray,              # (N,) float32
    gids: np.ndarray,             # (N,) int32
    group_rgb_table: np.ndarray,  # (max_gid + 1,) uint32
    gene_rgbs: np.ndarray,        # (N,) uint32
    is_at_homes: np.ndarray,      # (N,) bool
    in_swamps: np.ndarray,        # (N,) bool
    surplus_foods: np.ndarray,    # (N,) float32
    sprints: np.ndarray,          # (N,) float32
    reputations: np.ndarray,      # (N,) float32
    n_blobs: int,
) -> Tuple[float, int, int]:
    """
    High-throughput Numba kernel packing living blobs into the 8-column float32 WebSocket buffer.
    Copies kinematics, IDs, resolved group/gene RGB integers, and computes 7-bit flag bitmasks.
    Returns (rep_sum, outlaw_count, honorable_count).
    """
    outlaw_count = 0
    honorable_count = 0
    rep_sum = 0.0
    table_len = len(group_rgb_table)

    for i in range(n_blobs):
        gid = gids[i]
        rgb = (
            group_rgb_table[gid]
            if (0 <= gid < table_len and group_rgb_table[gid] != 0)
            else gene_rgbs[i]
        )
        rep = reputations[i]
        rep_sum += rep
        is_outlaw = rep <= -0.35
        is_honorable = rep >= 0.35
        if is_outlaw:
            outlaw_count += 1
        elif is_honorable:
            honorable_count += 1

        fl = 0
        if is_at_homes[i]:
            fl |= 1
        if in_swamps[i]:
            fl |= 2
        if surplus_foods[i] > 0.05:
            fl |= 4
        if sprints[i] > 0.35:
            fl |= 8
        if gid >= 0:
            fl |= 16
        if is_outlaw:
            fl |= 32
        elif is_honorable:
            fl |= 64

        out_buf[i, 0] = np.float32(xs[i])
        out_buf[i, 1] = np.float32(ys[i])
        out_buf[i, 2] = headings[i]
        out_buf[i, 3] = radii[i]
        out_buf[i, 4] = foods_today[i]
        out_buf[i, 5] = ids[i]
        out_buf[i, 6] = np.float32(rgb)
        out_buf[i, 7] = np.float32(fl)

    return rep_sum, outlaw_count, honorable_count


def warmup_all_jit_kernels() -> None:
    """Pre-warm all JIT kernels for both contiguous ('C') and strided ('A') layouts."""
    try:
        # 1. Neural Mutation
        _dw = np.zeros((1, 1), dtype=np.float32)
        _db = np.zeros((1,), dtype=np.float32)
        mutate_neural_layer_numba(_dw, _db, _dw, _db, False, 0.05, 0.1)
        mutate_neural_layer_numba(_dw, _db, _dw, _db, True, 0.05, 0.1)

        # 2. Forward GEMV: Both Contiguous ('C') and Strided ('A') Layouts
        w_c = np.zeros((96, 26, 24), dtype=np.float32)
        b_c = np.zeros((96, 24), dtype=np.float32)
        a_c = np.zeros((96, 26), dtype=np.float32)
        out_c = np.zeros((96, 24), dtype=np.float32)
        noise = np.zeros((0, 0), dtype=np.float32)
        parallel_mlp_layer_forward(a_c, w_c, b_c, out_c, 0, False, noise, False)
        parallel_mlp_layer_forward(out_c, np.zeros((96, 24, 6), dtype=np.float32), np.zeros((96, 6), dtype=np.float32), np.zeros((96, 6), dtype=np.float32), 0, True, noise, False)

        w_big = np.zeros((192, 26, 24), dtype=np.float32)
        b_big = np.zeros((192, 24), dtype=np.float32)
        a_big = np.zeros((192, 26), dtype=np.float32)
        parallel_mlp_layer_forward(a_big[::2], w_big[::2], b_big[::2], out_c, 0, False, noise, False)
        parallel_mlp_layer_forward(out_c, np.zeros((192, 24, 6), dtype=np.float32)[::2], np.zeros((192, 6), dtype=np.float32)[::2], np.zeros((96, 6), dtype=np.float32), 0, True, noise, False)

        # 3. Backprop Sensitivity Delta: Both Contiguous ('C') and Strided ('A') Layouts
        d_in = np.zeros((96, 6), dtype=np.float32)
        w_out_c = np.zeros((96, 24, 6), dtype=np.float32)
        act_prev = np.zeros((96, 24), dtype=np.float32)
        d_out = np.zeros((96, 24), dtype=np.float32)
        parallel_mlp_backprop_delta(w_out_c, d_in, act_prev, d_out, 0)
        w_out_big = np.zeros((192, 24, 6), dtype=np.float32)
        parallel_mlp_backprop_delta(w_out_big[::2], d_in, act_prev, d_out, 0)

        # 4. Rock CSR Grid & Market/Peer Filters
        dummy_pts = np.zeros(16, dtype=np.float32)
        dummy_f = np.zeros(2, dtype=np.float32)
        dummy_i = np.zeros(2, dtype=np.int32)
        batch_point_hits_rock_csr(dummy_pts, dummy_pts, 10.0, dummy_f, dummy_f, dummy_f, dummy_f, dummy_f, dummy_f, dummy_i, dummy_i, 1.0, 1, 1)

        # 5. Binary Blob Packing Kernel
        dummy_out = np.zeros((2, 8), dtype=np.float32)
        dummy_f64 = np.zeros(2, dtype=np.float64)
        dummy_f32 = np.zeros(2, dtype=np.float32)
        dummy_i32 = np.zeros(2, dtype=np.int32)
        dummy_u32 = np.zeros(2, dtype=np.uint32)
        dummy_bool = np.zeros(2, dtype=np.bool_)
        dummy_tab = np.zeros(4, dtype=np.uint32)
        pack_binary_blobs_kernel(
            dummy_out, dummy_f64, dummy_f64, dummy_f32, dummy_f32,
            dummy_f32, dummy_f32, dummy_i32, dummy_tab, dummy_u32,
            dummy_bool, dummy_bool, dummy_f32, dummy_f32, dummy_f32, 2
        )

        # 6. Full Simulation Engine (Spatial, Kinematics, Observation, Peer, Market)
        warmup_simulation_engine()
    except Exception:
        pass


def warmup_simulation_engine() -> None:
    """Pre-warm all 9 spatial, kinematics, observation, peer, and market kernels via a 4-tick mini-world."""
    from blob_sim.config import WorldConfig
    from blob_sim.world import SimulationWorld
    cfg = WorldConfig.from_preset("continent", initial_blobs=96, ticks_per_day=4)
    if cfg.num_rock_formations < 1:
        cfg.num_rock_formations = 1
    if cfg.num_swamps < 1:
        cfg.num_swamps = 1
    w_mini = SimulationWorld(world_cfg=cfg, seed=1)
    w_mini.step(4)
    w_mini.close()
    del w_mini
    import gc
    gc.collect()


warmup_all_jit_kernels()



