"""Load the open DFL/Sportec matches (Bassek et al. 2025, Scientific Data, CC-BY 4.0).

Source: https://springernature.figshare.com/articles/dataset/28196177 (seven Bundesliga and
2. Bundesliga matches). Unlike the Metrica samples, every ball frame carries the tracking
provider's BallStatus (1 in play, 0 dead) and BallPossession (1 home, 2 guest), the same kind
of labels the competition answers are derived from. Positions are pitch-centred metres and are
shifted to the 0..105 x 0..68 convention used elsewhere. Referees are skipped.
The files are large XML; they are parsed once and cached as compressed arrays. ElementTree
does not resolve external entities and Python's bundled expat limits entity expansion, so the
stdlib parser is acceptable for these published files.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .external_samples import ExternalMatch

DEFAULT_DFL_ROOT = Path(__file__).resolve().parents[1] / 'data/external/dfl-open'
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / 'data/cache/dfl'
SECTIONS = ('firstHalf', 'secondHalf')


@dataclass
class DflMatch:
    match: ExternalMatch
    labels: np.ndarray       # 'H' / 'A' / 'D' per frame; 'X' where the ball has no status
    match_id: str


def _match_info(path):
    root = ET.parse(path).getroot()
    general = root.find('.//General')
    pitch = root.find('.//Environment')
    teams = {team.get('TeamId'): team.get('Role') for team in root.iter('Team')}
    return {'home': general.get('HomeTeamId'), 'guest': general.get('GuestTeamId'), 'teams': teams,
            'pitch': (float(pitch.get('PitchX')), float(pitch.get('PitchY')))}


def _parse_positions(path):
    """Return {section: {'players': {(team, person): (N, x, y)}, 'ball': (N, x, y, poss, status)}}."""
    out = {s: {'players': {}, 'ball': None} for s in SECTIONS}
    current = None
    frames = []
    for event, element in ET.iterparse(path, events=('start', 'end')):
        tag = element.tag
        if event == 'start' and tag == 'FrameSet':
            current = (element.get('GameSection'), element.get('TeamId'), element.get('PersonId'))
            frames = []
        elif event == 'end' and tag == 'Frame':
            a = element.attrib
            frames.append((int(a['N']), float(a['X']), float(a['Y']),
                           int(a.get('BallPossession', 0)), int(a.get('BallStatus', -1))))
            element.clear()
        elif event == 'end' and tag == 'FrameSet':
            section, team, person = current
            if section in out and team != 'referee' and frames:
                data = np.asarray(frames, dtype=np.float64)
                if team.upper() == 'BALL':
                    out[section]['ball'] = data
                else:
                    out[section]['players'][(team, person)] = data[:, :3]
            element.clear()
    return out


def _cache_key(paths):
    digest = hashlib.sha256()
    for path in paths:
        stat = Path(path).stat()
        digest.update(f'{Path(path).name}:{stat.st_size}:{int(stat.st_mtime)}'.encode())
    return digest.hexdigest()[:16]


def load_dfl_match(info_path, positions_path, cache_dir=DEFAULT_CACHE):
    info = _match_info(info_path)
    match_id = Path(positions_path).stem.split('_')[-1]
    cache = Path(cache_dir) / f'{match_id}_{_cache_key([info_path, positions_path])}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            xy, home, ball, period, labels, pitch = (data[k] for k in ['xy', 'home', 'ball', 'period', 'labels', 'pitch'])
    else:
        parsed = _parse_positions(positions_path)
        slots = sorted({key for s in SECTIONS for key in parsed[s]['players']})
        home = np.array([info['teams'].get(team) == 'home' for team, _ in slots])
        length, width = info['pitch']
        xs, balls, periods, label_parts = [], [], [], []
        for index, section in enumerate(SECTIONS, start=1):
            ball = parsed[section]['ball']
            first, last = int(ball[:, 0].min()), int(ball[:, 0].max())
            n = last - first + 1
            xy = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
            for s, key in enumerate(slots):
                data = parsed[section]['players'].get(key)
                if data is not None:
                    rows = data[:, 0].astype(int) - first
                    xy[rows, s] = np.c_[data[:, 1] + length / 2, data[:, 2] + width / 2]
            b = np.full((n, 2), np.nan)
            rows = ball[:, 0].astype(int) - first
            b[rows] = np.c_[ball[:, 1] + length / 2, ball[:, 2] + width / 2]
            label = np.full(n, 'X', dtype='<U1')
            label[rows] = np.where(ball[:, 4] == 0, 'D', np.where(ball[:, 3] == 1, 'H',
                                   np.where(ball[:, 3] == 2, 'A', 'X')))
            xs.append(xy); balls.append(b); periods.append(np.full(n, index, np.int8)); label_parts.append(label)
        xy, ball, period, labels = (np.concatenate(xs), np.concatenate(balls), np.concatenate(periods),
                                    np.concatenate(label_parts))
        pitch = np.array(info['pitch'])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, xy=xy, home=home, ball=ball, period=period, labels=labels, pitch=pitch)
    return DflMatch(ExternalMatch(f"DFL_{match_id.split('-')[-1]}", xy, home, ball, period, [], 25.0, tuple(float(v) for v in pitch)),
                    labels, match_id)


def load_dfl_matches(root=DEFAULT_DFL_ROOT, cache_dir=DEFAULT_CACHE):
    root = Path(root)
    positions = sorted(root.glob('DFL_04_03_positions_raw_observed_*.xml'))
    if not positions:
        raise FileNotFoundError(f'No DFL position files in {root}; see solution/README.md for the download')
    out = []
    for path in positions:
        match_id = path.stem.split('_')[-1]
        info = next(root.glob(f'DFL_02_01_matchinformation_*_{match_id}.xml'))
        out.append(load_dfl_match(info, path, cache_dir))
    return out
