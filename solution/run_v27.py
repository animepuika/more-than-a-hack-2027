"""V27: fixed-setting, whole-match test of adding SkillCorner possession training.

No BAR-ATH scoring or output changes. Checkpoints permit safe resumption.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .run import ROOT, sha256, write_labels
from .run_v23 import INCUMBENT, cluster_difference, counts_score, spell_counts
from .split_cleanup import tidy

PROTOCOL = """# V27 fixed whole-match possession-training comparison
2026-10-27. Written before training or reading this experiment's outcomes.

Purpose: isolate additional possession training from dead-ball and decoder changes.
Reference: Metrica training only. Candidate: same Metrica plus five SkillCorner
matches selected by ascending SHA256('20271027:'+match_id), before loading labels.
Other 15 SkillCorner matches are held out from ALL fitting in this run. Primary
evaluation: all 3 Metrica and 7 DFL matches; a Metrica target is excluded from
both training pools. DFL never trains either possession model here. These games
have been used historically: this is not a pristine confirmation dataset.

Both arms use unchanged V12 turnover and V10 gap model parameters and row strides.
Turnover training: recorded + 25%-hidden live-ball copies. Gap training: recorded.
SkillCorner ball-mask seed index is 3 + its position in sorted SkillCorner names.
No parameter search. V12 switch cost is 0.5 s for every match in BOTH arms;
V10's existing 1 s switch/2 s endpoint anchor settings remain unchanged.

Evaluate full clean AND realistic test_edge copies of all 10 primary matches,
plus full recorded versions of all held-out SkillCorner matches. Primary result
is realistic test_edge. Input features and masks are identical across arms.
Primary final D mask is frozen from each saved leave-one-match-out V16 prediction
after existing 1/3/4 s cleanup. SkillCorner final D is frozen from its V1 output
after that same cleanup (no learned two-camera D input exists there). Decode H/A
with that shared final D mask, run existing gap fallback, then only 1 s flip
cleanup; do not re-decide D from either arm's new team labels. This isolates H/A
and differs from the historical pipeline's ordering: report both the shared-
decoder reference and historical cleaned predictions, never conflate them.

Targets: Metrica and SkillCorner provider/event labels unchanged; primary DFL
uses the previously frozen sub-second-touch cleanup. Repeat all scores on original
DFL labels. Actual final prediction D masks must match exactly across arms.
Report official Spell F1 across all five IoUs, per match/provider, whole-match
bootstrap, and live-team agreement broken down by reliable ball/long gaps.
Subset frame agreement is diagnostic, not whole-game Spell F1.

Pass rules, applied only after all planned matches finish:
P1 primary realistic pooled Spell F1 gain >=0.01 over the shared-decoder reference;
P2 positive median per-primary-match gain AND >=80% positive paired whole-match
   bootstrap draws (2000, seed 20271027);
P3 no provider pooled loss >0.01: Metrica and DFL realistic, SkillCorner recorded;
P4 primary realistic aggregate gain >=0 on raw provider targets too;
P5 candidate primary score >= historical current cleanup score +0.01;
P6 identical D masks, valid full-length H/A/D/X outputs, unchanged inactive labels,
   exclusion of each evaluation target from all training, stable sources/inputs;
P7 historical realistic primary score reproduces 0.5225879682179342 within 1e-12.
Clean copies are sensitivity diagnostics, not a second chance to pass.

No BAR-ATH public/private labels, no official public scoring, no automatic
submission replacement, commit or push. If all pass, a separate whole-BAR-ATH
application still needs its own predeclared checks. Audio/camera ownership are
kept out of this controlled training comparison; their actual dual-view inputs
are absent on the practice games. Existing evidence remains in V26.
"""


def training_pool(target, metrica, skillcorner, mixed):
    return [n for n in list(metrica)+(list(skillcorner) if mixed else []) if n != target]


def ownership_cleanup(labels, period, fps):
    result = tidy(labels, period, fps, flip_s=1, short_dead_s=0, dead_gap_s=0)
    if not np.array_equal(result == 'D', labels == 'D'):
        raise AssertionError('Ownership cleanup changed dead-ball mask')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'data/runs/twentyseventh_trial')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    out=args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()): raise ValueError('Output must be under data/')
    sources_hash={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}
    scroot=ROOT/'data/external/skillcorner'
    scids=sorted(p.name for p in scroot.iterdir() if p.is_dir() and (p/f'{p.name}_match.json').exists())
    ordered=sorted(scids,key=lambda s:hashlib.sha256(('20271027:'+s).encode()).hexdigest())
    if len(scids)!=20: raise ValueError('Expected 20 SkillCorner matches before fixing split')
    split={'training_skillcorner':['SC_'+n for n in ordered[:5]],'evaluation_skillcorner':['SC_'+n for n in ordered[5:]],
           'all_skillcorner_sorted':['SC_'+n for n in scids]}
    saved_path=ROOT/'data/runs/bench10_tuning/tuning_labels.npz'
    lock={'split':split,'source_sha256':sources_hash,'saved_predictions_sha256':sha256(saved_path)}
    if args.resume:
        if json.loads((out/'lock.json').read_text())!=lock: raise ValueError('Resume inputs/code differ')
        if (out/'PROTOCOL.md').read_text()!=PROTOCOL: raise ValueError('Protocol differs')
    else:
        out.mkdir(exist_ok=False)
        (out/'PROTOCOL.md').write_text(PROTOCOL,encoding='utf8')
        (out/'lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf8')
        (out/'models').mkdir(); (out/'predictions').mkdir(); (out/'rows').mkdir()
    print('Protocol/split locked. Loading external whole matches.',flush=True)
    from .run_bench10 import load_sources, get_bundle, features_dict
    from .feature_cache import fingerprint
    from .gap_fallback import long_gap_mask
    from .gap_model import MODEL_SETTINGS as GAP_SETTINGS, training_rows as gap_rows, home_probability as gap_probability
    from .run_v13 import combine
    from .run_v16 import controlled_possession
    from .turnover_model import fit as fit_turn, training_rows as turn_rows, touch_features, home_probability as turn_probability, decode
    from evaluation.possession_metrics import read_label_csv, validate_submission
    data=load_sources(('metrica','dfl','skillcorner'))
    by_name={s['name']:s for s in data}
    metrica=[s['name'] for s in data if s['kind']=='metrica']
    core=[s for s in data if s['kind'] in ('metrica','dfl')]
    if len(core)!=10 or len(metrica)!=3: raise ValueError('Expected 3 Metrica + 7 DFL')
    input_hashes={s['name']:{'tracking':fingerprint(s['match']),
                            'labels':hashlib.sha256(np.ascontiguousarray(s['labels']).tobytes()).hexdigest()} for s in data}
    input_path=out/'data_fingerprints.json'
    if args.resume and json.loads(input_path.read_text())!=input_hashes: raise ValueError('Dataset contents differ')
    if not args.resume: input_path.write_text(json.dumps(input_hashes,indent=2)+'\n',encoding='utf8')
    quality=json.loads((ROOT/'solution/config_v7.json').read_text())
    saved=np.load(saved_path,allow_pickle=False)
    # Prepare compact training rows once. Labels of evaluation games never enter
    # these tables; each Metrica target is excluded again at model fitting.
    training={}
    train_names=metrica+split['training_skillcorner']
    for name in train_names:
        s=by_name[name]
        idx=metrica.index(name) if name in metrica else 3+split['all_skillcorner_sorted'].index(name)
        parts=[]
        for version in ('recorded','ball25'):
            print('Training features:',name,version,flush=True)
            m,b=get_bundle(s,idx,version)
            bf=(b['bf_home'],b['bf_away'])
            parts.append(turn_rows(touch_features(features_dict(b),m.ball_xy_m,m.period,bf),s['labels']))
            if version=='recorded': gap=gap_rows(bf,s['labels'])
            del m,b,bf
        training[name]={'turn':parts,'gap':gap}
        gc.collect()

    model_refs={}
    for held in [None]+metrica:
        for arm,mixed in [('reference',False),('mixed',True)]:
            pool=training_pool(held,metrica,split['training_skillcorner'],mixed)
            key=arm+'_'+(held or 'all')
            path=out/'models'/f'{key}.joblib'
            meta_path=out/'models'/f'{key}.json'
            if args.resume and path.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text())
                if meta['training_matches']!=pool or meta['sha256']!=sha256(path): raise ValueError('Model checkpoint differs')
                print('Reusing model:',key,flush=True)
            else:
                print('Fitting',key,'on',len(pool),'whole matches.',flush=True)
                start=time.monotonic()
                with threadpool_limits(limits=4):
                    turn=fit_turn([p for name in pool for p in training[name]['turn']])
                    gx=pd.concat([training[name]['gap'][0] for name in pool],ignore_index=True)
                    gy=np.concatenate([training[name]['gap'][1] for name in pool])
                    gap=HistGradientBoostingClassifier(**GAP_SETTINGS).fit(gx,gy)
                joblib.dump({'turn':turn,'gap':gap},path)
                meta={'training_matches':pool,'sha256':sha256(path),'seconds':time.monotonic()-start}
                meta_path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
                del turn,gap,gx,gy
                print('Fitted',key,'seconds',round(meta['seconds'],1),flush=True)
            model_refs[key]=meta
            gc.collect()
    del training
    gc.collect()
    rows=[]
    jobs=[(s,i,condition,version) for i,s in enumerate(core) for condition,version in [('clean','recorded'),('realistic','test_edge')]]
    jobs += [(by_name[n],0,'recorded','recorded') for n in split['evaluation_skillcorner']]
    loaded={}
    for s,i,condition,version in jobs:
        name=s['name']; row_path=out/'rows'/f'{condition}_{name}.json'
        if args.resume and row_path.exists():
            row=json.loads(row_path.read_text())
            if any(sha256(out/rel)!=digest for rel,digest in row['prediction_sha256'].items()): raise ValueError('Prediction checkpoint differs')
            rows.append(row); print('Reusing scored row',condition,name,flush=True); continue
        print('Predicting whole match:',condition,name,flush=True)
        m,b=get_bundle(s,i,version)
        feats=features_dict(b); active=m.period>0
        historical=tidy(saved[f'{condition}|{name}|v16_gated_cal'],m.period,m.fps,**INCUMBENT) if s['kind']!='skillcorner' else tidy(b['v1'],m.period,m.fps,**INCUMBENT)
        dead=(historical=='D') & active
        raw=s['labels']; truth=controlled_possession(raw,m.period,m.fps) if s['kind']=='dfl' else raw
        tables=touch_features(feats,m.ball_xy_m,m.period,(b['bf_home'],b['bf_away']))
        variants={}
        membership={}
        for arm in ['reference','mixed']:
            key=arm+'_'+(name if name in metrica else 'all')
            membership[arm]=model_refs[key]['training_matches']
            if name in membership[arm]: raise ValueError('Evaluation match entered training')
            if key not in loaded:
                loaded[key]=joblib.load(out/'models'/f'{key}.joblib')
            model=loaded[key]
            with threadpool_limits(limits=4):
                pt=turn_probability(model['turn'],tables)
                pg=gap_probability(model['gap'],(b['bf_home'],b['bf_away']))
            pred=decode(pt,m.period,dead,.5,m.fps)
            pred=ownership_cleanup(combine(pred,pg,feats,m.period,quality,m.fps),m.period,m.fps)
            if not np.array_equal(pred=='D',dead) or not np.all(pred[~active]=='X'): raise ValueError('Frozen mask/inactive output changed')
            variants[arm]=pred
        row={'name':name,'kind':s['kind'],'condition':condition,'training_membership':membership,'prediction_sha256':{},'metrics':{}}
        for arm,pred in variants.items():
            path=out/'predictions'/f'{condition}_{name}_{arm}.csv'; write_labels(path,pred)
            error=validate_submission(read_label_csv(path),len(m.period))
            if error: raise ValueError(error)
            row['prediction_sha256'][path.relative_to(out).as_posix()]=sha256(path)
        for definition,target in [('controlled',truth),('raw',raw)]:
            counts={arm:spell_counts(target,pred) for arm,pred in variants.items()}
            row['metrics'][definition]={'counts':{arm:c.tolist() for arm,c in counts.items()},'spell_f1':{arm:float(counts_score(c)) for arm,c in counts.items()}}
            if s['kind']!='skillcorner':
                row['metrics'][definition]['historical_counts']=spell_counts(target,historical).tolist()
        long=long_gap_mask(feats['ball_reliable']==1,m.period,round(2*m.fps))
        row['live_team_diagnostics']={}
        for label,mask in [('reliable_ball',feats['ball_reliable']==1),('long_ball_gap',long)]:
            use=active & mask & np.isin(truth,['H','A']) & ~dead
            row['live_team_diagnostics'][label]={'frames':int(use.sum()),**{arm:float(np.mean(pred[use]==truth[use])) if use.any() else None for arm,pred in variants.items()}}
        row_path.write_text(json.dumps(row,indent=2)+'\n',encoding='utf8')
        rows.append(row)
        print('  Spell F1',row['metrics']['controlled']['spell_f1'],flush=True)
        del m,b,feats,tables,variants
        gc.collect()

    def aggregate(selected,definition='controlled'):
        count={arm:sum(np.array(r['metrics'][definition]['counts'][arm]) for r in selected) for arm in ['reference','mixed']}
        score={arm:float(counts_score(c)) for arm,c in count.items()}
        return {'spell_f1':score,'gain':score['mixed']-score['reference'],'counts':{arm:c.tolist() for arm,c in count.items()}}
    primary=[r for r in rows if r['condition']=='realistic']
    if len(primary)!=10 or len(rows)!=35: raise ValueError('Incomplete planned benchmark')
    groups={f'{condition}/{kind}':aggregate([r for r in rows if r['condition']==condition and (kind=='all' or r['kind']==kind)])
            for condition,kinds in [('realistic',['all','metrica','dfl']),('clean',['all','metrica','dfl']),('recorded',['skillcorner'])] for kind in kinds}
    primary_score=groups['realistic/all']
    raw_score=aggregate(primary,'raw')
    paired=cluster_difference([r['metrics']['controlled']['counts']['mixed'] for r in primary],
                              [r['metrics']['controlled']['counts']['reference'] for r in primary])
    median=float(np.median([r['metrics']['controlled']['spell_f1']['mixed']-r['metrics']['controlled']['spell_f1']['reference'] for r in primary]))
    historical=float(counts_score(sum(np.array(r['metrics']['controlled']['historical_counts']) for r in primary)))
    stable=all(sha256(ROOT/p)==h for p,h in sources_hash.items()) and sha256(saved_path)==lock['saved_predictions_sha256']
    stable=stable and all(fingerprint(s['match'])==input_hashes[s['name']]['tracking'] and hashlib.sha256(np.ascontiguousarray(s['labels']).tobytes()).hexdigest()==input_hashes[s['name']]['labels'] for s in data)
    rules={'P1_primary_gain':primary_score['gain']>=.01,
           'P2_match_consistency':median>0 and paired['share_first_better']>=.80,
           'P3_provider_safety':all(groups[k]['gain']>=-.01 for k in ['realistic/metrica','realistic/dfl','recorded/skillcorner']),
           'P4_raw_labels':raw_score['gain']>=0,
           'P5_beats_historical':primary_score['spell_f1']['mixed']>=historical+.01,
           'P6_integrity':stable,'P7_historical_reproduced':abs(historical-.5225879682179342)<=1e-12}
    report={'experiment':'V27 whole-match fixed possession-training comparison','protocol_sha256':sha256(out/'PROTOCOL.md'),
            'split':split,'groups':groups,'primary_raw_labels':raw_score,'primary_cluster_bootstrap':paired,
            'primary_median_match_gain':median,'historical_primary_cleanup':historical,'rules':rules,
            'decision':'eligible_for_whole_bar_ath_trial' if all(rules.values()) else 'reject_training_change',
            'per_match':rows,'model_checkpoints':model_refs,'source_sha256':sources_hash,
            'public_ground_truth_read':False,'submission_changed':False,
            'limitations':['Previously inspected data; no pristine confirmation set.',
                           'Practice D masks are frozen; full BAR-ATH multimodal pipeline is not measured.',
                           'Provider labels differ; raw target sensitivity is included.',
                           'Mask simulation uses known external labels; same conditions for both arms.']}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    (ROOT/'solution/results/first_trial_v27.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ['groups','primary_cluster_bootstrap','primary_median_match_gain','historical_primary_cleanup','rules','decision']},indent=2),flush=True)


if __name__=='__main__': main()
