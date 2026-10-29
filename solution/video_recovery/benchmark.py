"""Controlled experimental ownership rebuild. Never trains or reads answer labels.

Use the scientific .venv Python, not the separate GPU/media environment.
First verify cached ownership model probabilities and exact incumbent labels.
Only then apply recovered ball observations, freezing final incumbent D/X.
"""
from pathlib import Path
import argparse,json,sys,time,platform
HERE=Path(__file__).resolve().parent
from .config import ROOT, OUT
import joblib,numpy as np,pandas as pd,sklearn
from threadpoolctl import threadpool_limits
from games import GAME,GAMES,asset_paths,half_video_blocks
from solution.data import load_tracking
from solution.external_samples import ExternalMatch
from solution.gap_model import ballfree_features
from solution.make_submission import CLEANUP
from solution.run import sha256,write_labels
from solution.run_v6 import _period_keys
from solution.run_v12 import SWITCH_COST_S
from solution.run_v13 import combine
from solution.split_cleanup import tidy
from solution.turnover_model import touch_features,home_probability,decode
from solution.v7_features import build_features
from solution.v8_stoppages import build_evidence


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery',type=Path,default=OUT/'recovery.npz')
    parser.add_argument('--out',type=Path,default=OUT/'benchmark')
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args(argv)
    out=args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Outputs must stay inside ignored data/')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a fresh output directory; no prediction overwrite')
    out.mkdir(parents=True,exist_ok=True)
    report={'started_unix':time.time(),'status':'running','baseline_only':args.baseline_only,
            'public_answers_read':False,'model_refit':False,'candidate_promoted':False,
            'environment':{'python':platform.python_version(),'numpy':np.__version__,
                           'pandas':pd.__version__,'sklearn':sklearn.__version__},
            'checks':{},'input_hashes':{}}

    def save():
        report['elapsed_seconds']=time.time()-report['started_unix']
        (out/'benchmark_build.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')

    def stop(reason):
        report['status']='blocked';report['blocker']=reason;save()
        raise RuntimeError(reason)

    try:
        frozen=OUT/'inputs'
        paths=asset_paths()
        inputs={'model':frozen/'turnover.joblib',
                'model_metadata':frozen/'model.json',
                'p12':frozen/'p12.npy',
                'p10':frozen/'p10.npy',
                'v2':frozen/'v2.csv',
                'v12':frozen/'v12.csv',
                'v13':frozen/'v13.csv',
                'cutaway':frozen/'cutaway.csv',
                'incumbent':frozen/'incumbent.csv',
                'quality':ROOT/'solution/config_v7.json','v8config':ROOT/'solution/config_v8.json',
                'tracking_metadata':Path(paths['atd_meta']),'tracking':Path(paths['atd_raw']),
                'script':Path(__file__)}
        inputs.update({'solution/'+str(p.relative_to(ROOT/'solution')):p for p in sorted((ROOT/'solution').rglob('*.py'))})
        inputs.update({'protocol/'+p.name:p for p in sorted(HERE.glob('*PROTOCOL*.md'))})
        if not args.baseline_only:inputs['recovery']=args.recovery.resolve()
        for key,path in inputs.items():
            if not path.exists():stop('Required input missing: '+key+' '+str(path))
            report['input_hashes'][key]={'path':str(path),'sha256':sha256(path)}
        meta=json.loads(inputs['model_metadata'].read_text())
        if meta['sha256']!=report['input_hashes']['model']['sha256']:
            stop('Cached model differs from its saved metadata')
        report['training_matches']=meta.get('training_matches')
        if GAME in meta.get('training_matches',[]):stop('Target match entered cached training')
        quality=json.loads(inputs['quality'].read_text())
        v8config=json.loads(inputs['v8config'].read_text())
        n,fps,pitch=(GAMES[GAME][k] for k in ('n_video_frames','fps','pitch'))
        period,active=_period_keys(n,half_video_blocks(GAME))
        print('Loading supplied tracking and frozen probabilities.',flush=True)
        tracking=load_tracking(paths['atd_meta'],paths['atd_raw'],n,fps,pitch,retain_tracks=True)
        original=ExternalMatch(GAME,tracking.track_xy_m,tracking.home_slots,tracking.ball_xy_m,period,[],fps,pitch)
        labels={key:pd.read_csv(inputs[key]).label.to_numpy(dtype='<U1')
                for key in ('v2','v12','v13','cutaway','incumbent')}
        if any(len(lab)!=n for lab in labels.values()):stop('Frozen label row count mismatch')
        p12_saved=np.load(inputs['p12']);p10=np.load(inputs['p10'])
        if p12_saved.shape!=(n,) or p10.shape!=(n,):stop('Frozen probability shape mismatch')
        loaded_model=joblib.load(inputs['model'])
        model=loaded_model['turn'] if isinstance(loaded_model,dict) else loaded_model
        print('Building original full-match V7 and ball-free features.',flush=True)
        features,_=build_features(original,quality)
        ballfree=ballfree_features(original.player_xy_m,original.home_slots,period,pitch,fps)

        def probability(match,built):
            tables=touch_features(built,match.ball_xy_m,period,ballfree)
            p=np.full(n,.5)
            with threadpool_limits(limits=4):
                p[active]=home_probability(model,(tables[0][active],tables[1][active]))
            if not np.isfinite(p).all():stop('Nonfinite inferred probabilities')
            return p

        inferred=probability(original,features)
        difference=float(np.max(np.abs(inferred[active]-p12_saved[active])))
        report['checks']['original_p12_max_abs_difference']=difference
        if difference>1e-6:stop('Cached model does not reproduce original V12 probabilities within 1e-6; no delta workaround')

        def pipeline(match,p):
            v12=decode(p,period,(labels['v2']=='D')&active,SWITCH_COST_S,fps)
            evidence=build_evidence(match,v8config,quality)
            v13=combine(v12,p10,evidence,period,quality,fps)
            cut=v13.copy();cut[(labels['cutaway']=='D')&active]='D'
            cleaned=tidy(cut,period,fps,**CLEANUP)
            return v12,v13,cleaned

        print('Verifying original decode, gap combination and full cleanup.',flush=True)
        baseline12,baseline13,baseline=pipeline(original,inferred)
        for key,array in (('v12',baseline12),('v13',baseline13),('incumbent',baseline)):
            count=int(np.count_nonzero(array!=labels[key]))
            report['checks'][key+'_different_frames']=count
            if count:stop('Original '+key+' labels do not reproduce exactly')
        write_labels(out/'baseline_pred.csv',baseline)
        exact_bytes=sha256(out/'baseline_pred.csv')==sha256(inputs['incumbent'])
        report['checks']['baseline_file_byte_identical']=exact_bytes
        if not exact_bytes:stop('Original labels match but frozen CSV bytes differ; no recovery export')
        report['checks']['baseline_reproduced']=True
        print('Original baseline reproduces byte-for-byte.',flush=True)
        if args.baseline_only:
            report['status']='baseline_verified';save();return

        archive=np.load(args.recovery,allow_pickle=False)
        if 'merged_ball' not in archive.files:stop('Recovery NPZ lacks merged_ball')
        ball=np.asarray(archive['merged_ball'])
        if ball.shape!=(n,2):stop('Recovered ball must have shape (199319,2)')
        observed=np.isfinite(tracking.ball_xy_m).any(axis=1)
        if not np.array_equal(ball[observed],tracking.ball_xy_m[observed],equal_nan=True):
            stop('Recovery altered an existing finite or partial original observation')
        new=np.isnan(tracking.ball_xy_m).all(axis=1)&np.isfinite(ball).all(axis=1)
        invalid=(~observed)&(~np.isnan(ball).all(axis=1))&(~np.isfinite(ball).all(axis=1))
        if invalid.any():stop('Recovery introduced partial/nonfinite candidate coordinates')
        if np.any(new&~active):stop('Recovery introduced observations outside active periods')
        report['recovered_frames']=int(new.sum())
        recovered=ExternalMatch(GAME,original.player_xy_m,original.home_slots,ball,period,[],fps,pitch)
        print('Building recovered full-match V7 features; ball-free table remains fixed.',flush=True)
        recovered_features,_=build_features(recovered,quality)
        p_new=probability(recovered,recovered_features)
        _,_,candidate=pipeline(recovered,p_new)
        incumbent=labels['incumbent'];fixed=np.isin(incumbent,['D','X'])
        unexpected_dead=(~fixed)&(~np.isin(candidate,['H','A']))
        report['cleanup_ownership_fallback_frames']=int(unexpected_dead.sum())
        candidate[unexpected_dead]=incumbent[unexpected_dead]
        candidate[fixed]=incumbent[fixed]
        report['checks']['final_dead_mask_identical']=bool(np.array_equal(candidate=='D',incumbent=='D'))
        report['checks']['final_x_mask_identical']=bool(np.array_equal(candidate=='X',incumbent=='X'))
        report['changed_ownership_frames']=int(np.count_nonzero(candidate!=incumbent))
        if not all(report['checks'][k] for k in ('final_dead_mask_identical','final_x_mask_identical')):
            stop('Final incumbent dead/X masks changed')
        stable=all(sha256(Path(v['path']))==v['sha256'] for v in report['input_hashes'].values())
        report['checks']['input_hashes_stable']=stable
        if not stable:stop('An input changed during benchmark rebuild')
        write_labels(out/'predictions.csv',candidate)
        np.save(out/'recovered_p12.npy',p_new)
        report['prediction_sha256']=sha256(out/'predictions.csv')
        report['status']='experimental_predictions_ready';save()
        print(json.dumps({k:report[k] for k in ('status','recovered_frames','changed_ownership_frames','cleanup_ownership_fallback_frames')},indent=2),flush=True)
    except Exception as error:
        if report['status']=='running':
            report['status']='failed';report['error']=repr(error);save()
        raise


if __name__=='__main__':main()
