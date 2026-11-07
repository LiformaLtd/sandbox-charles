"""Copy explicit read-only research inputs into this standalone visual sandbox."""
import argparse,hashlib,json,shutil,wave
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from cheek_transfer import build_cheek_transfer

OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,
      152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
UP=[191,80,81,82,13,312,311,310,415];DOWN=[95,88,178,87,14,317,402,318,324]


def triangles(edges):
    # Same fixed-topology clique construction used in the standalone mapping probe.
    adj=defaultdict(set)
    for a,b in edges:
        if a!=b:adj[a].add(b);adj[b].add(a)
    seen=set();ts=[]
    for a,b in edges:
        if a==b:continue
        for c in sorted(adj[a]&adj[b]):
            t=tuple(sorted([a,b,c]))
            if t not in seen and max(t)<478:ts.append(t);seen.add(t)
    def links(t):return [tuple(sorted([t[a],t[b]])) for a,b in [(0,1),(1,2),(2,0)]]
    removed=set()
    while True:
        count=Counter(e for i,t in enumerate(ts) if i not in removed for e in links(t))
        score=[sum(max(count[e]-2,0) for e in links(t)) if i not in removed else 0 for i,t in enumerate(ts)]
        if not score or max(score)==0:break
        removed.add(int(np.argmax(score)))
    return np.array([t for i,t in enumerate(ts) if i not in removed],np.uint16).reshape(-1).tolist()


def write(path,value):path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--avatars',type=Path,required=True)
    parser.add_argument('--maps',type=Path,required=True)
    parser.add_argument('--local-maps',type=Path,required=True)
    parser.add_argument('--source-assets',type=Path,required=True)
    args=parser.parse_args();app=Path(__file__).resolve().parents[1];out=app/'static/assets'
    out.mkdir(parents=True,exist_ok=True)
    results=json.loads((args.maps/'results.json').read_text());reg=np.load(args.maps/'source-registration.npz')
    if results['protocol']['model_sha256']!='b4a275acf78b982fb6ec6fc55333c5cf25613e72b6d9b3d80e95d6e74ac6261d':
        raise ValueError('Unexpected mapping model contract')
    names={a['id']:a.get('name',a['id']) for a in json.loads((args.avatars/'index.json').read_text())['avatars']}
    catalog=[];fingerprints={};skipped=[]
    for result in results['faces']:
        face=result['id'];folder=out/'avatars'/face;folder.mkdir(parents=True,exist_ok=True)
        mesh_file=args.avatars/face/'face-meshes.json';mesh=json.loads(mesh_file.read_text())
        neutral=next(v for v in mesh['visemes'] if v['id']=='neutral')
        src=args.avatars/face/neutral['sourceFile']
        if not src.exists() and '_old' in face:
            skipped.append(dict(id=face,reason='Historical snapshot has no matching neutral texture'))
            continue
        shutil.copy2(src,folder/'neutral.webp')
        with Image.open(src) as image:
            size=image.size;image.thumbnail((160,160));image.save(folder/'thumbnail.webp')
        model=np.load(args.maps/(face+'-mapping.npz'))
        reference=model['reference'].astype(np.float32)
        center=model['frame_center'];width=float(model['frame_width']);rotation=model['frame_rotation']
        reference_image=reference@rotation.T*width+center
        box=reference_image[np.array(OVAL)]
        xmin,ymin=box.min(0);xmax,ymax=box.max(0);w=xmax-xmin;h=ymax-ymin
        # Include hair above the landmark forehead and preserve physical image aspect.
        crop=dict(x=float(xmin-w*.38),y=float(ymin-h*.3),w=float(w*1.76),h=float(h*1.55))
        midpoint=reference_image[np.array(OVAL)].mean(0)
        ring=(reference_image[np.array(OVAL)]-midpoint)*1.8+midpoint
        indices=triangles(mesh['faceMeshEdges'])
        for i,a in enumerate(OVAL):
            b=OVAL[(i+1)%len(OVAL)];ra=478+i;rb=478+(i+1)%len(OVAL)
            indices.extend([a,b,ra,b,rb,ra])
        linear=dict(kind='linear',reference=reference.ravel().tolist(),basis=model['basis'].astype(np.float32).ravel().tolist(),
            sourceReference=model['source_reference'].astype(np.float32).tolist())
        write(folder/'linear.json',linear)
        local_file=args.local_maps/(face+'-local-map.npz')
        if not local_file.exists():raise ValueError('Missing explicit local residual map: '+str(local_file))
        local=np.load(local_file)
        local_map=dict(kind='local',reference=local['reference'].astype(np.float32).ravel().tolist(),
            sourceReference=local['source_reference'].astype(np.float32).tolist(),basis=local['prior'].astype(np.float32).ravel().tolist(),
            featureBasis=local['feature_basis'].astype(np.float32).ravel().tolist(),featureCount=local['feature_basis'].shape[0],
            training=local['train_features'].astype(np.float32).ravel().tolist(),exampleCount=len(local['train_features']),
            solution=local['solution'].astype(np.float32).ravel().tolist(),bandwidth=float(local['bandwidth']))
        write(folder/'local.json',local_map)
        data=dict(id=face,name=names.get(face,face),image='/assets/avatars/'+face+'/neutral.webp',
            imageSize=size,sourceUV=np.array(neutral['landmarks'])[:,:2].ravel().tolist(),
            ring=ring.ravel().tolist(),triangles=indices,crop=crop,
            frame=dict(center=center.tolist(),width=width,rotation=rotation.ravel().tolist()),
            cheek=build_cheek_transfer(reg,model,mesh['faceMeshEdges']),
            fitHash=sha(mesh_file),calibration='Experimental — not validated for production',
            contactGap=[.003]*9)
        write(folder/'mesh.json',data)
        catalog.append(dict(id=face,name=names.get(face,face),thumbnail='/assets/avatars/'+face+'/thumbnail.webp',
            url='/assets/avatars/'+face+'/mesh.json'))
        fingerprints[face]=dict(mesh=sha(mesh_file),neutralTexture=sha(src))
    study=args.source_assets/'compact-v4-confirmation';rows=json.loads((study/'manifest.json').read_text())
    picks=[next(r for r in rows if r['stem']=='test-supertonic-m3-'+f'{i:03}') for i in [0,4,8,12,16,20]]
    samples=[];(out/'speech').mkdir(exist_ok=True)
    labels=['M sounds','B sounds','P sounds','Rounded vowels','Connected speech','Conversation']
    for row,label in zip(picks,labels):
        pcm=np.fromfile(row['audio'],'<f4');file=out/'speech'/(row['stem']+'.wav')
        with wave.open(str(file),'wb') as wav:
            wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000)
            wav.writeframes(np.rint(np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())
        samples.append(dict(id=row['stem'],label=label,text=row['text'],url='/assets/speech/'+file.name,duration=len(pcm)/16000))
    model=args.source_assets/'compact-v4-validation/v4-candidate.onnx'
    expected=results['protocol']['model_sha256']
    if sha(model)!=expected:raise ValueError('Pinned model checksum mismatch')
    (out/'models').mkdir(exist_ok=True);shutil.copy2(model,out/'models/lf-sta-v4.onnx')
    speech_dir=out/'lf-tts';speech_dir.mkdir(exist_ok=True)
    for name in ['model.q8.gguf','tokenizer.json','config.json','easton.safetensors']:
        shutil.copy2(args.source_assets/'phonon'/name,speech_dir/name)
    upper,lower=np.array([[9013,10613],[941,10433],[8859,10434],[2963,3726],[1439,1475],
        [5980,7505],[6093,18810],[263,18809],[17362,13358]]).T
    n,b=reg['source_neutral'],reg['source_basis']
    contact=dict(rest=(n[upper,1]-n[lower,1]).tolist(),basis=(b[upper,1,:]-b[lower,1,:]).ravel().tolist())
    write(out/'catalog.json',dict(avatars=catalog,samples=samples,excludedHistoricalFits=skipped,
        model=dict(url='/assets/models/lf-sta-v4.onnx',sha256=expected,bytes=model.stat().st_size),contact=contact))
    write(out/'provenance.json',dict(avatars=fingerprints,model=expected,sourceMaps=str(args.maps.resolve()),localMaps=str(args.local_maps.resolve())))
    print(f'Prepared {len(catalog)} actual avatar fits, 6 speech clips, the pinned model and local speech assets in {out}')
    for entry in skipped:print('Excluded',entry['id']+':',entry['reason'])


if __name__=='__main__':main()
