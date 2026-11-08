"""Freeze natural prefix/suffix recordings and byte-aligned controlled input streams."""
import re,numpy as np
from common import *
from geometry import prepare as rig_prepare

def main():
 records=[];manifest_hashes={}
 for p in sorted((SCALE/'corpus/base-shards').glob('*/manifest.json')):
  manifest_hashes[str(p)]=sha(p);records += [r for r in json.loads(p.read_text()) if r['split']=='train']
 voices=sorted({r['engine']+':'+r['voice'] for r in records});assert len(voices)==8
 norm=lambda t:' '.join(re.findall('[a-z]+',t.lower()))
 order=lambda r,salt:hashlib.sha256((salt+r['text']+r['voice']).encode()).hexdigest()
 selected=[];seen=set()
 for voice in voices:
  eligible=sorted([r for r in records if r['engine']+':'+r['voice']==voice and r['samples']>=64000],key=lambda r:order(r,'history-suffix37:'))
  picks=[]
  for r in eligible:
   if norm(r['text']) not in seen:picks.append(r);seen.add(norm(r['text']))
   if len(picks)==2:break
  assert len(picks)==2,voice
  selected += picks
 cases=[];audio_root=OUT/'audio';audio_root.mkdir(parents=True,exist_ok=True)
 for voice in voices:
  prefix_rows=sorted([r for r in records if r['engine']+':'+r['voice']==voice and r['samples']>=64000 and norm(r['text']) not in seen],key=lambda r:order(r,'history-prefix37:'))
  prefixes=[];prefix_texts=set()
  for r in prefix_rows:
   if norm(r['text']) not in prefix_texts:prefixes.append(r);prefix_texts.add(norm(r['text']))
   if len(prefixes)==2:break
  assert len(prefixes)==2
  for r in [r for r in selected if r['engine']+':'+r['voice']==voice]:
   suffix=np.fromfile(r['audio'],'<f4')[:64000].copy();assert sha(r['audio'])==r['sha256_audio'];suffix[:320]*=np.linspace(0,1,320,dtype=np.float32)
   variants={'cold':suffix,'silence':np.concatenate((np.zeros(64000,np.float32),suffix))}
   source={r['stem']:dict(audio=r['audio'],sha256=r['sha256_audio'],text=r['text'])}
   for name,p in zip(['speech-a','speech-b'],prefixes):
    assert sha(p['audio'])==p['sha256_audio'];prefix=np.fromfile(p['audio'],'<f4')[:64000].copy();prefix[-640:-320]*=np.linspace(1,0,320,dtype=np.float32);prefix[-320:]=0
    variants[name]=np.concatenate((prefix,suffix));source[p['stem']]=dict(audio=p['audio'],sha256=p['sha256_audio'],text=p['text'])
   case_id=f'case-{len(cases):02}';paths={}
   for name,pcm in variants.items():
    path=audio_root/(case_id+'-'+name+'.f32');freeze(path,pcm.astype('<f4').tobytes());paths[name]=dict(path=str(path),sha256=sha(path),prefix_samples=0 if name=='cold' else 64000)
   for relative in range(0,48001,8000):
    windows=[pcm[(0 if name=='cold' else 64000)+relative:(0 if name=='cold' else 64000)+relative+16000] for name,pcm in variants.items()]
    assert all(len(w)==16000 and np.array_equal(w,windows[0]) for w in windows)
   cases.append(dict(id=case_id,engine=r['engine'],voice=r['voice'],gender=r['gender'],text=r['text'],audio=paths,source_recordings=source))
 rig=rig_prepare();freeze(OUT/'cases.json',cases)
 sources=[HERE/'common.py',HERE/'prepare.py',HERE/'projector.py',HERE/'run.py',HERE/'summarize.py',APP/'docs/plans/teacher-history-compatibility.md',APP/'scripts/research/lip_surface/geometry.py']
 freeze(OUT/'protocol.json',dict(base_manifest_sha256=manifest_hashes,cases_sha256=sha(OUT/'cases.json'),cases=16,voices=voices,seeds=[0,1,2,3],histories=['cold','silence','speech-a','speech-b'],bins=[[0,.25],[.25,.75],[.75,1.25],[1.25,2.25],[2.25,3.25]],primary=[.25,3.25],surface_vertices=len(rig['ids']),baseline_sha256=sha(BASELINE),source_sha256={str(p.relative_to(APP)):sha(p) for p in sources},teacher_source_sha256={p.name:sha(p) for p in (TEACHER/'app/audio2face_v3').glob('*.py')},assets_sha256={n:sha(ASSETS/n) for n in ['model_data_James.npz','bs_skin_James.npz','bs_skin_config_James.json']}))
 for p in sources:freeze(OUT/'sources/frozen'/p.relative_to(APP),p.read_bytes())
 print('Frozen16 distinct suffixes ×4 histories ×4 seeds; identical common windows checked.',flush=True)
if __name__=='__main__':main()
