"""Experimental causal short-horizon recovery over supplied missing-ball gaps."""
from pathlib import Path
import os,json,time,hashlib
from .config import ROOT,OUT,WEIGHT_SHA256
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_settings');os.environ['YOLO_AUTOINSTALL']='false'
import cv2,numpy as np,torch
from ultralytics import YOLO,settings
from .tracker import BallIdentityTracker,_project
settings.update({'sync':False});torch.set_num_threads(2)
from games import asset_paths
VIDEO=Path(os.environ.get('MTAH_TACTICAL_VIDEO',asset_paths()['video']))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    begin=time.time();weights=OUT/'football-ball-detection.pt';reference=OUT/'reference.npz'
    paths=[Path(__file__),Path(__file__).with_name('tracker.py'),Path(__file__).with_name('config.py'),weights,reference,VIDEO]
    if digest(weights)!=WEIGHT_SHA256:raise ValueError('Pinned model hash mismatch')
    lock={str(p):digest(p) for p in paths}
    if (OUT/'recovery_lock.json').exists() and json.loads((OUT/'recovery_lock.json').read_text())!=lock:raise RuntimeError('Recovery inputs changed')
    save('recovery_lock.json',lock)
    ref=np.load(reference);ball=ref['ball'];pixels=ref['pixel'];H=ref['H'];N=len(ball)
    period=np.zeros(N,int);period[11185:90652]=1;period[110808:190198]=2
    missing=np.isnan(ball).all(axis=1);z=np.r_[False,missing,False].astype(int)
    starts=np.flatnonzero(np.diff(z)==1);ends=np.flatnonzero(np.diff(z)==-1)
    model=YOLO(str(weights));assert model.names=={0:'ball'}
    cap=cv2.VideoCapture(str(VIDEO));checkpoint=OUT/'recovery_gaps';checkpoint.mkdir(exist_ok=True)
    allrows=[];gapstats=[]
    for index,(a,b) in enumerate(zip(starts,ends)):
        a,b=int(a),int(b);path=checkpoint/f'{index:03}.json'
        if path.exists():item=json.loads(path.read_text());allrows.extend(item['rows']);gapstats.append(item['summary']);continue
        seed=a-1;rows=[];reason=None
        if seed<0 or period[a]<=0 or period[seed]!=period[a]:reason='no_same_half_seed'
        elif not np.isfinite(pixels[seed]).all() or not (0<=pixels[seed,0]<1920 and 0<=pixels[seed,1]<1080):reason='no_in_image_seed'
        else:
            tracker=BallIdentityTracker(pixels[seed],H[seed]);lastxy=ball[seed].copy();lastframe=seed
            cap.set(cv2.CAP_PROP_POS_FRAMES,a)
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=a:raise RuntimeError('Wrong seek')
            for i in range(a,min(b,a+24)):
                if period[i]!=period[a] or tracker.lost:break
                ok,img=cap.read()
                if not ok:raise RuntimeError('Decode failure')
                warp=H[i]@np.linalg.inv(tracker.H);expected=_project(tracker.position+tracker.velocity,warp)
                if expected is None:reason='invalid_prediction';break
                x=int(np.clip(round(expected[0])-320,0,1280));y=int(np.clip(round(expected[1])-320,0,440))
                pred=model.predict(img[y:y+640,x:x+640],imgsz=640,conf=.1,iou=.5,max_det=50,classes=[0],device=0,quantize=16,rect=False,verbose=False,save=False)
                if len(pred)!=1:raise RuntimeError('Unexpected batch count')
                boxes=pred[0].boxes.data.cpu().numpy()[:,:5].astype(float)
                boxes[:,[0,2]]+=x;boxes[:,[1,3]]+=y
                if not np.isfinite(boxes).all():raise RuntimeError('Invalid detection')
                selected=tracker.step(boxes,H[i]);xy=None;why='tracker_abstained'
                if selected is not None:
                    pixel=(selected[:2]+selected[2:4])/2;q=np.linalg.inv(H[i])@np.r_[pixel,1.]
                    projected=q[:2]/q[2]*[105.3,68]
                    if not np.isfinite(projected).all() or (projected< -3).any() or (projected>np.array([105.3,68])+3).any():why='outside_pitch_or_invalid'
                    elif np.isfinite(lastxy).all() and np.linalg.norm(projected-lastxy)*25/(i-lastframe)>70:why='implausible_projected_speed'
                    else:xy=projected.tolist();lastxy=projected;lastframe=i;why='observed_recovery'
                rows.append({'frame':i,'boxes':boxes.tolist(),'selected':None if selected is None else selected.tolist(),
                             'xy':xy,'reason':why,'tracker_lost':tracker.lost,'roi':[x,y,x+640,y+640]})
        summary={'gap_index':index,'start':a,'end':b,'missing_frames':b-a,'attempted':len(rows),
                 'recovered':sum(r['xy'] is not None for r in rows),'skip_reason':reason}
        item={'summary':summary,'rows':rows};path.write_text(json.dumps(item,indent=2)+'\n')
        allrows.extend(rows);gapstats.append(summary)
        save('recovery_status.json',{'completed_gaps':index+1,'total_gaps':len(starts),'attempted_frames':len(allrows),
             'recovered_frames':sum(r['xy'] is not None for r in allrows),'elapsed_seconds':time.time()-begin,'complete':False})
        if index%10==0:print(summary,flush=True)
    cap.release();candidate=np.full_like(ball,np.nan)
    for row in allrows:
        if row['xy'] is not None:candidate[row['frame']]=row['xy']
    fill=missing&np.isfinite(candidate).all(axis=1);merged=ball.copy();merged[fill]=candidate[fill]
    original_finite=np.isfinite(ball)
    assert np.array_equal(merged[original_finite],ball[original_finite])
    np.savez_compressed(OUT/'recovery.npz',merged_ball=merged,candidate_ball=candidate,filled=fill)
    result={'complete':True,'gaps':gapstats,'attempted_frames':len(allrows),'recovered_frames':int(fill.sum()),
            'original_missing_frames':int(missing.sum()),'fraction_missing_recovered':float(fill.sum()/missing.sum()),
            'public_recovered_frames':int(fill[11185:18685].sum()),'observed_coordinates_preserved':True,
            'source_hashes_stable':all(digest(p)==h for p,h in lock.items()),'elapsed_seconds_this_invocation':time.time()-begin,
            'candidate_promoted':False,'output_sha256':digest(OUT/'recovery.npz')}
    save('recovery_results.json',result);save('recovery_status.json',{k:v for k,v in result.items() if k!='gaps'})
    print(json.dumps({k:v for k,v in result.items() if k!='gaps'},indent=2),flush=True)
if __name__=='__main__':main()
