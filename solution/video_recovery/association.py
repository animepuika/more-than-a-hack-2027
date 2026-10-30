"""Sparse endpoint graph with exact alternative-path marginals, no target labels."""
import numpy as np


def project_pixel(H, pixel):
    try:
        q = np.linalg.solve(np.asarray(H, float), np.r_[pixel, 1.])
        if not np.isfinite(q).all() or abs(q[2]) < 1e-9: return None
        p = q[:2]/q[2]*[105.3, 68.]
        return p if np.isfinite(p).all() and (p >= -3).all() and (p <= [108.3, 71.]).all() else None
    except np.linalg.LinAlgError:
        return None


def center(box):
    b = np.asarray(box, float)
    return (b[:2]+b[2:4])/2


def associate(frames, boxes, H, start_frame, start_pixel, start_H, end_frame, end_pixel, end_H):
    frames = np.asarray(frames, int); m = len(frames)
    if not m or len(boxes) != m or len(H) != m:
        raise ValueError('Invalid sampled frame counts')
    if not start_frame < frames[0] or not frames[-1] < end_frame or not (np.diff(frames) > 0).all():
        raise ValueError('Invalid time ordering')
    selected = [None]*m
    p0 = project_pixel(start_H, start_pixel); pe = project_pixel(end_H, end_pixel)
    if p0 is None or pe is None:
        return {'selected': selected, 'reason': 'invalid_endpoint', 'path_nodes': 0}
    times = [start_frame]; ranks = [-1]; xy = [p0]; pixels = [np.asarray(start_pixel)]
    nodeboxes = [None]; costs = [0.]; buckets = []
    for t, frame in enumerate(frames):
        bucket = []
        b = np.asarray(boxes[t], float).reshape(-1, 5)
        valid = np.isfinite(b).all(1) & (b[:, 4] >= .1) & (b[:, 4] <= 1.) & (b[:, 2:4] > b[:, :2]).all(1)
        b = b[valid]; b = b[np.argsort(-b[:, 4], kind='stable')[:8]]
        for box in b:
            pixel = center(box); point = project_pixel(H[t], pixel)
            if point is None: continue
            bucket.append(len(times)); times.append(int(frame)); ranks.append(t); xy.append(point)
            pixels.append(pixel); nodeboxes.append(box.tolist()); costs.append(-float(np.log(box[4])))
        buckets.append(bucket)
    end = len(times); times.append(end_frame); ranks.append(m); xy.append(pe)
    pixels.append(np.asarray(end_pixel)); nodeboxes.append(None); costs.append(0.)
    times = np.asarray(times); ranks = np.asarray(ranks); xy = np.asarray(xy); costs = np.asarray(costs)
    parents = [np.empty(0, int)]; edges = [np.empty(0)]
    for j in range(1, end+1):
        if j == end:
            possible = np.arange(end)
        else:
            lo = max(1, int(np.searchsorted(times[:j], times[j]-25, side='left')))
            hi = int(np.searchsorted(times[:j], times[j], side='left'))
            possible = np.r_[0, np.arange(lo, hi)].astype(int)
        dt = (times[j]-times[possible])/25.
        dist = np.linalg.norm(xy[j]-xy[possible], axis=1)
        allowed = (dt > 0) & (dist <= 70*dt+2)
        possible = possible[allowed]; dt = dt[allowed]; dist = dist[allowed]
        edge = 2*np.maximum(ranks[j]-ranks[possible]-1, 0)+.25*(dist/(20*dt+2))**2
        parents.append(possible); edges.append(edge)
    forward = np.full(end+1, np.inf); prev = np.full(end+1, -1, int); forward[0] = 0.
    for j in range(1, end+1):
        if not len(parents[j]): continue
        values = forward[parents[j]]+edges[j]+costs[j]; k = int(np.argmin(values))
        forward[j] = values[k]; prev[j] = parents[j][k]
    best = forward[end]
    if not np.isfinite(best): return {'selected': selected, 'reason': 'no_path', 'path_nodes': 0}
    backward = np.full(end+1, np.inf); backward[end] = 0.
    for j in range(end, 0, -1):
        values = edges[j]+costs[j]+backward[j]
        backward[parents[j]] = np.minimum(backward[parents[j]], values)
    bypass = np.full(m, np.inf)
    for j in range(1, end+1):
        values = forward[parents[j]]+edges[j]+costs[j]+backward[j]
        for parent, value in zip(parents[j], values):
            lo, hi = max(0, int(ranks[parent])+1), min(m, int(ranks[j]))
            if hi > lo and np.isfinite(value): bypass[lo:hi] = np.minimum(bypass[lo:hi], value)
    path = []; j = int(prev[end])
    while j > 0:
        path.append(j); j = int(prev[j])
    margins = [None]*m
    for j in path:
        t = int(ranks[j]); others = [k for k in buckets[t] if np.linalg.norm(pixels[k]-pixels[j]) > 20]
        alternate = min([bypass[t], *[forward[k]+backward[k] for k in others]])
        margin = float(alternate-best); margins[t] = margin if np.isfinite(margin) else None
        if margin >= .5: selected[t] = nodeboxes[j]
    return {'selected': selected, 'margins': margins, 'reason': 'associated', 'path_nodes': len(path),
            'path_cost': float(best), 'candidate_nodes': end-1,
            'edge_count': sum(len(p) for p in parents)}


def deduplicate(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 5)
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:4] > boxes[:, :2]).all(1)
    valid &= (boxes[:, 4] >= .1) & (boxes[:, 4] <= 1)
    boxes = boxes[valid]
    order = np.argsort(-boxes[:, 4], kind='stable')
    kept = []
    while len(order) and len(kept) < 50:
        i = int(order[0]); kept.append(i); order = order[1:]
        if not len(order):
            break
        lo = np.maximum(boxes[i, :2], boxes[order, :2])
        hi = np.minimum(boxes[i, 2:4], boxes[order, 2:4])
        inter = np.maximum(hi-lo, 0).prod(1)
        area = (boxes[:, 2:4]-boxes[:, :2]).prod(1)
        iou = inter / (area[i]+area[order]-inter)
        order = order[iou <= .5]
    return boxes[kept]
