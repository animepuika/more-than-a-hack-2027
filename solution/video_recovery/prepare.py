"""Prepare local baseline inputs and supplied image projections, without answers."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from games import GAME,GAMES,asset_paths
from solution.data import load_tracking
from solution.run import sha256
from .config import ROOT,OUT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--baseline-build',type=Path,help='make_submission --save-video-inputs output folder')
    group.add_argument('--from-existing-runs',action='store_true',help='Import pre-existing V12/V20/V27 local checkpoints')
    args=parser.parse_args();dest=OUT/'inputs'
    if dest.exists() and any(dest.iterdir()):raise ValueError('Inputs already exist; choose a fresh output folder')
    dest.mkdir(parents=True,exist_ok=True)
    if args.baseline_build:
        source=args.baseline_build.resolve()/'video_inputs'
        mapping={name:source/name for name in ('turnover.joblib','model.json','p12.npy','p10.npy','v2.csv','v12.csv','v13.csv','cutaway.csv','incumbent.csv')}
    else:
        runs=ROOT/'data/runs'
        mapping={'turnover.joblib':runs/'twentyseventh_trial/models/reference_all.joblib',
                 'model.json':runs/'twentyseventh_trial/models/reference_all.json',
                 'p12.npy':runs/'twelfth_trial/v12_p_home.npy','p10.npy':runs/'tenth_trial/v10_p_home.npy',
                 'v2.csv':runs/'seventh_trial/predictions/v2_shape_motion.csv',
                 'v12.csv':runs/'twelfth_trial/predictions/v12_turnover.csv',
                 'v13.csv':runs/'thirteenth_trial/predictions/v13_combined.csv',
                 'cutaway.csv':runs/'twentieth_trial/predictions/v20_cutaway.csv',
                 'incumbent.csv':runs/'twentysecond_trial/predictions/v20_clean.csv'}
    if not all(p.exists() for p in mapping.values()):raise ValueError('Missing baseline inputs; build with --save-video-inputs first')
    for name,path in mapping.items():shutil.copy2(path,dest/name)
    paths=asset_paths();g=GAMES[GAME];n=g['n_video_frames'];pitch=np.array(g['pitch'])
    tracking=load_tracking(paths['atd_meta'],paths['atd_raw'],n,g['fps'],tuple(pitch))
    hom=pd.read_csv(paths['homography'])
    if not np.array_equal(hom.frame.to_numpy(),np.arange(n)):raise ValueError('Homography row mismatch')
    H=hom[[f'h{i}{j}' for i in range(3) for j in range(3)]].to_numpy().reshape(-1,3,3)
    q=np.einsum('nij,nj->ni',H,np.c_[tracking.ball_xy_m/pitch,np.ones(n)])
    ok=np.isfinite(tracking.ball_xy_m).all(axis=1)&np.isfinite(q).all(axis=1)&(np.abs(q[:,2])>1e-9)
    pixel=np.full_like(tracking.ball_xy_m,np.nan);pixel[ok]=q[ok,:2]/q[ok,2,None]
    np.savez_compressed(OUT/'reference.npz',ball=tracking.ball_xy_m,pixel=pixel,H=H)
    manifest={'source_hashes':{str(p):sha256(p) for p in mapping.values()},
              'supplied_hashes':{k:sha256(paths[k]) for k in ('atd_meta','atd_raw','homography')},
              'input_hashes':{p.name:sha256(p) for p in dest.iterdir()},'reference_sha256':sha256(OUT/'reference.npz'),
              'public_answers_read':False}
    (OUT/'prepare.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared',OUT)
if __name__=='__main__':main()
