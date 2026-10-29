"""Portable locations. Set MTAH_VIDEO_RECOVERY_OUT before invoking a module."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('MTAH_VIDEO_RECOVERY_OUT',ROOT/'data/runs/video_recovery')).resolve()
if not OUT.is_relative_to((ROOT/'data').resolve()):
    raise ValueError('Video recovery outputs must remain inside ignored data/')
OUT.mkdir(parents=True,exist_ok=True)
WEIGHT_SHA256='678fbad05134f19c5094cb8d273812ec9c6691228180d46832551ecf99ed2912'
