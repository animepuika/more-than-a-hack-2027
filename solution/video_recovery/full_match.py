"""Expanded full-match video experiment; actual detector boxes only.

Run with ``python -m solution.video_recovery.full_match --help``.
The recorded experiment did not beat the existing short-video candidate.
"""
from pathlib import Path
import os, json, time, hashlib, argparse
import numpy as np
from . import association as association_module
from .association import associate, center, project_pixel, deduplicate
from .config import ROOT, OUT as BASE_OUT, WEIGHT_SHA256
from games import GAME, half_video_blocks, asset_paths

BEST_SHA = 'd2048553ec2ad4538369a5f786c24531d2d95439c0bf6f3a9102bf1a0aa240f5'
TILES = [(x, y, x+640, y+640) for y in (0, 440) for x in (0, 640, 1280)]


def _media(video, weights):
    os.environ['YOLO_CONFIG_DIR'] = str(OUT/'ultralytics_settings')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    if model.names != {0: 'ball'}: raise ValueError('Expected the pinned ball-only model')
    return cv2, model, cv2.VideoCapture(str(video))


def _output_path(value):
    value = Path(value).resolve()
    if not value.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Outputs must stay under ignored data/')
    value.mkdir(parents=True, exist_ok=True)
    return value

def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path); temp = path.with_suffix('.part')
    temp.write_text(json.dumps(obj, indent=2)+'\n'); temp.replace(path)


def inference(model, items):
    """items: (frame, imagecrop, ROI); all returned centres are actual boxes."""
    if not items: return []
    result = model.predict([v[1] for v in items], imgsz=640, conf=.1, iou=.5, max_det=50,
                           classes=[0], device=0, quantize=16, rect=False, verbose=False, save=False)
    if len(result) != len(items): raise RuntimeError('Microbatch count differs')
    output = []
    for pred, (i, _, roi) in zip(result, items):
        b = pred.boxes.data.cpu().numpy()[:, :5].astype(float)
        b[:, [0, 2]] += roi[0]; b[:, [1, 3]] += roi[1]
        if not np.isfinite(b).all() or (b[:, 2:4] <= b[:, :2]).any(): raise RuntimeError('Invalid detection')
        output.append((i, b))
    return output


def normalized_pixel(H, norm):
    q = H@np.r_[norm, 1.]
    return q[:2]/q[2] if np.isfinite(q).all() and abs(q[2]) > 1e-9 else None


def main(argv=None):
    global OUT, REF, WEIGHTS, VIDEO, SHORT, BEST, BASELINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=BASE_OUT/'full_match')
    parser.add_argument('--reference', type=Path, default=BASE_OUT/'reference.npz')
    parser.add_argument('--weights', type=Path, default=BASE_OUT/'football-ball-detection.pt')
    parser.add_argument('--short-recovery', type=Path, default=BASE_OUT/'recovery.npz')
    parser.add_argument('--short-baseline', type=Path, default=BASE_OUT/'benchmark')
    parser.add_argument('--video', type=Path, default=Path(os.environ.get('MTAH_TACTICAL_VIDEO', asset_paths()['video'])))
    args = parser.parse_args(argv)
    OUT = _output_path(args.out)
    REF, WEIGHTS, VIDEO, SHORT = (p.resolve() for p in (args.reference, args.weights, args.video, args.short_recovery))
    BASELINE = args.short_baseline.resolve()
    BEST = BASELINE/'predictions.csv'
    began = time.time()
    if digest(BEST) != BEST_SHA:
        raise RuntimeError('Best video baseline not reproduced exactly')
    baseline = json.loads((BASELINE/'benchmark_build.json').read_text())
    if baseline['status'] != 'experimental_predictions_ready' or not baseline['checks']['baseline_file_byte_identical']:
        raise RuntimeError('Baseline checks not complete')
    if digest(WEIGHTS) != WEIGHT_SHA256: raise RuntimeError('Wrong detector')
    sources = [Path(__file__), Path(association_module.__file__),
               ROOT/'solution/results/video_recovery_full_match.json',
               REF, WEIGHTS, VIDEO, SHORT, BEST,
               ROOT/'solution/video_recovery/benchmark.py', BASELINE/'benchmark_build.json']
    lock = {str(p): digest(p) for p in sources}
    if (OUT/'recovery_lock.json').exists() and json.loads((OUT/'recovery_lock.json').read_text()) != lock:
        raise RuntimeError('Recovery inputs changed')
    save(OUT/'recovery_lock.json', lock)
    r = np.load(REF); source_ball = r['ball']; pixels = r['pixel']; H = r['H']; n = len(source_ball)
    short = np.load(SHORT)['merged_ball']; merged = short.copy(); initial_fixed = np.isfinite(short).any(1)
    original_missing = np.isnan(source_ball).all(1)
    active = np.zeros(n, bool); periods = np.zeros(n, int)
    for half, a, b in half_video_blocks(GAME): active[a:b+1] = True; periods[a:b+1] = half
    missing = original_missing & active
    split = np.diff(np.r_[False, missing, False].astype(int)); starts = np.flatnonzero(split == 1); ends = np.flatnonzero(split == -1)
    (OUT/'samples').mkdir(exist_ok=True); (OUT/'gaps').mkdir(exist_ok=True)
    model = cap = cv2 = None
    summaries = []; live = {'phase': 'recovery', 'total_gaps': len(starts)}

    def progress(phase, gap_index, **extra):
        save(OUT/'status.json', {**live, 'phase': phase, 'gap_index': gap_index, **extra,
                                'elapsed_seconds': time.time()-began})

    for index, (a, b) in enumerate(zip(starts, ends)):
        a, b = int(a), int(b); finish = OUT/'gaps'/f'{index:03d}.json'
        if finish.exists():
            item = json.loads(finish.read_text())
            if item['start'] != a or item['end'] != b: raise RuntimeError('Gap checkpoint differs')
            for row in item['added']: merged[row['frame']] = row['xy']
            summaries.append(item['summary']); continue
        reason = None
        if a <= 0 or b >= n or periods[a-1] != periods[a] or periods[b] != periods[a]: reason = 'no_two_same_half_endpoints'
        elif not (np.isfinite(pixels[[a-1, b]]).all() and (pixels[[a-1, b]] >= 0).all()
                  and (pixels[[a-1, b], 0] < 1920).all() and (pixels[[a-1, b], 1] < 1080).all()): reason = 'endpoints_not_both_in_image'
        if reason:
            summary = {'gap': index, 'half': int(periods[a]), 'start': a, 'end': b, 'frames': b-a,
                       'skip_reason': reason, 'added': 0, 'sampled_frames': 0, 'dense_frames': 0}
            save(finish, {'start': a, 'end': b, 'added': [], 'summary': summary}); summaries.append(summary); continue
        if model is None: cv2, model, cap = _media(VIDEO, WEIGHTS)
        sample_frames = list(range(a, b, 5)); sample_file = OUT/'samples'/f'{index:03d}.json'
        if sample_file.exists():
            sampled = json.loads(sample_file.read_text())
            if sampled['frames'] != sample_frames: raise RuntimeError('Sample checkpoint differs')
        else:
            boxes = {i: [] for i in sample_frames}; batch = []; cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != a: raise RuntimeError('Wrong sampling seek')
            for i in range(a, b):
                ok, image = cap.read()
                if not ok: raise RuntimeError('Sampling decode failed')
                if (i-a) % 5: continue
                for x, y, x2, y2 in TILES: batch.append((i, image[y:y2, x:x2], [x, y, x2, y2]))
                if len(batch) >= 12:
                    for frame, found in inference(model, batch): boxes[frame].extend(found.tolist())
                    batch = []
                if (i-a) % 250 == 0: progress('sampling', index, gap_start=a, gap_frames=b-a, frames_decoded=i-a+1)
            for frame, found in inference(model, batch): boxes[frame].extend(found.tolist())
            sampled = {'frames': sample_frames, 'boxes': [deduplicate(boxes[i]).tolist() for i in sample_frames]}
            save(sample_file, sampled)
        progress('association', index, gap_start=a, gap_frames=b-a)
        graph = associate(sample_frames, sampled['boxes'], H[sample_frames], a-1, pixels[a-1], H[a-1], b, pixels[b], H[b])
        sparse = []
        for i, box in zip(sample_frames, graph['selected']):
            if box is None or initial_fixed[i]: continue
            xy = project_pixel(H[i], center(box))
            if xy is not None: merged[i] = xy; sparse.append({'frame': i, 'xy': xy.tolist(), 'box': box, 'kind': 'sparse'})
        # Crop hints only; never export an interpolated position.
        anchor_frames = [a-1, *[i for i in range(a, b) if np.isfinite(merged[i]).all()], b]
        anchor_xy = {i: (source_ball[i] if i in (a-1, b) else merged[i]).copy()/[105.3, 68.] for i in anchor_frames}
        hints = {}
        for left, right in zip(anchor_frames[:-1], anchor_frames[1:]):
            if right-left > 10: continue
            for i in range(left+1, right):
                if initial_fixed[i] or np.isfinite(merged[i]).all(): continue
                xy = anchor_xy[left]+(anchor_xy[right]-anchor_xy[left])*(i-left)/(right-left)
                pixel = normalized_pixel(H[i], xy)
                if pixel is not None: hints[i] = pixel
        dense = []; dense_boxes = {}; batch = []

        def accept(items):
            for i, found in inference(model, items):
                dense_boxes[i] = found.tolist()
                if not len(found): continue
                centers = (found[:, :2]+found[:, 2:4])/2
                distance = np.linalg.norm(centers-hints[i], axis=1); use = np.flatnonzero(distance <= 25)
                if not len(use): continue
                j = use[int(np.argmin(distance[use]/25-.1*found[use, 4]))]
                xy = project_pixel(H[i], centers[j])
                if xy is not None:
                    merged[i] = xy; dense.append({'frame': i, 'xy': xy.tolist(), 'box': found[j].tolist(), 'kind': 'dense'})

        if hints:
            progress('dense_search', index, gap_start=a, gap_frames=b-a, frames_decoded=0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != a: raise RuntimeError('Wrong dense seek')
            for i in range(a, b):
                ok, image = cap.read()
                if not ok: raise RuntimeError('Dense decode failed')
                if i not in hints: continue
                point = hints[i]; x = int(np.clip(round(point[0])-320, 0, 1280)); y = int(np.clip(round(point[1])-320, 0, 440))
                batch.append((i, image[y:y+640, x:x+640], [x, y, x+640, y+640]))
                if len(batch) >= 12: accept(batch); batch = []
                if (i-a) % 250 == 0: progress('dense_search', index, gap_start=a, gap_frames=b-a, frames_decoded=i-a+1)
            accept(batch)
        added = sparse+dense
        summary = {'gap': index, 'half': int(periods[a]), 'start': a, 'end': b, 'frames': b-a,
                   'skip_reason': None, 'added': len(added), 'sampled_frames': len(sample_frames), 'dense_frames': len(hints),
                   'sparse_accepted': len(sparse), 'dense_accepted': len(dense), 'graph': {k: v for k, v in graph.items() if k != 'selected'}}
        save(finish, {'start': a, 'end': b, 'added': added, 'summary': summary, 'dense_boxes': dense_boxes})
        summaries.append(summary)
        progress('gap_complete', index, completed_gaps=len(summaries), additional_coordinates_so_far=int((np.isfinite(merged).all(1) & ~initial_fixed).sum()))
        print('GAP', index+1, '/', len(starts), 'length', b-a, 'added', len(added), flush=True)
    if cap is not None: cap.release()
    new = original_missing & ~initial_fixed & np.isfinite(merged).all(1)
    speed = np.linalg.norm(np.diff(merged, axis=0), axis=1)*25
    bad = np.flatnonzero(np.isfinite(speed) & (speed > 70))
    drop = np.unique(np.r_[bad[new[bad]], (bad+1)[new[bad+1]]])
    merged[drop] = np.nan
    new = original_missing & ~initial_fixed & np.isfinite(merged).all(1)
    assert np.array_equal(merged[initial_fixed], short[initial_fixed], equal_nan=True)
    observed = np.isfinite(source_ball).any(1)
    assert np.array_equal(merged[observed], source_ball[observed], equal_nan=True)
    assert not (new & ~active).any()
    total_filled = original_missing & np.isfinite(merged).all(1)
    np.savez_compressed(OUT/'recovery.npz', merged_ball=merged, filled=total_filled, added=new,
                        source_ball=source_ball, initial_short_ball=short)
    stable = all(digest(p) == h for p, h in lock.items()); assert stable
    result = {'complete': True, 'additional_coordinates': int(new.sum()), 'original_missing': int(original_missing.sum()),
              'active_missing': int(missing.sum()), 'total_recovered_coordinates': int(total_filled.sum()),
              'new_by_half': {str(h): int((new & (periods == h)).sum()) for h in (1, 2)},
              'speed_screen_dropped': len(drop), 'sampled_frames': sum(s['sampled_frames'] for s in summaries),
              'dense_frames': sum(s['dense_frames'] for s in summaries), 'gaps': summaries,
              'observations_and_short_recovery_preserved': True, 'source_hashes_stable': stable,
              'ready_for_integration': bool(new.sum() >= 500), 'public_answers_read': False,
              'elapsed_seconds': time.time()-began, 'recovery_sha256': digest(OUT/'recovery.npz')}
    save(OUT/'recovery_results.json', result); progress('recovery_complete', len(starts), **{k: v for k, v in result.items() if k != 'gaps'})
    print(json.dumps({k: v for k, v in result.items() if k != 'gaps'}, indent=2), flush=True)


if __name__ == '__main__': main()
