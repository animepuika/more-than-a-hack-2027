"""Fetch publisher-linked ball-only weights; inspect without unpickling."""
from pathlib import Path
import urllib.request,zipfile,hashlib,json,pickletools,re
from .config import OUT,WEIGHT_SHA256
URL='https://drive.usercontent.google.com/download?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V&export=download&confirm=t'
path=OUT/'football-ball-detection.pt'
if not path.exists():
    req=urllib.request.Request(URL,headers={'User-Agent':'local-research'})
    with urllib.request.urlopen(req,timeout=120) as response,path.with_suffix('.part').open('wb') as out:
        for b in iter(lambda:response.read(1024*1024),b''):out.write(b)
    if not zipfile.is_zipfile(path.with_suffix('.part')):raise RuntimeError('Not torch ZIP; refuse execution')
    path.with_suffix('.part').replace(path)
with path.open('rb') as file:
    if hashlib.file_digest(file,'sha256').hexdigest()!=WEIGHT_SHA256:raise ValueError('Publisher weight hash changed; refuse model load')
with zipfile.ZipFile(path) as archive:
    entry=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
    ops=list(pickletools.genops(archive.read(entry)))
strings=[str(a) for o,a,_ in ops if o.name in {'BINUNICODE','SHORT_BINUNICODE','UNICODE'}]
metadata={}
for i,(op,a,_) in enumerate(ops):
    if op.name in {'BINUNICODE','SHORT_BINUNICODE','UNICODE'} and a in {'date','version','license','model','imgsz','epochs'}:
        following=[(o,v) for o,v,_ in ops[i+1:i+5] if o.name not in {'BINPUT','LONG_BINPUT','MEMOIZE'}]
        if following and following[0][0].name in {'BINUNICODE','SHORT_BINUNICODE','UNICODE','BININT','BININT1','BININT2'}:
            v=following[0][1]
            if a!='model' or re.fullmatch(r'yolov\d+[a-z0-9]*\.pt',str(v)):metadata[a]=v
report={'url':URL,'publisher':'https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh',
        'dataset_version_source':'https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg/dataset/2',
        'dataset_version_date':'2024-07-25','dataset_license':'CC BY 4.0','binary_training_linkage_independently_verified':False,
        'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
        'metadata':metadata,'architecture_strings':[s for s in strings if re.fullmatch(r'yolov\d+[a-z0-9]*\.(pt|yaml)',s)],
        'types':sorted(set(str(a) for o,a,_ in ops if o.name=='GLOBAL')),
        'ball_class_string_present':'ball' in strings,'upstream_code_imported':False}
(OUT/'weights.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
