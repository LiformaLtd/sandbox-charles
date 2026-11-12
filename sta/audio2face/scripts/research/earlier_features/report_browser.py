"""Summarize independent browser trials without treating timing noise as a win."""
from common import *
import numpy as np

def main():
    result={}
    for arm in ['v4','late','early']:
        timing=[];parity=[];memory=[];allocation=[]
        for trial in range(1,4):
            rows=json.loads((OUT/f'browser/parity/run-{trial}.json').read_text());take=[r for r in rows if r['model']==arm and r['variant']=='fixed'];timing.append(next(r['median_ms'] for r in take if r['name']=='phonon-packed'));parity.extend(r['max'] for r in take)
            record=json.loads((OUT/f'memory/run-{trial}'/(arm+'.json')).read_text());memory.append(record['peak']/2**20);allocation.append(record['allocation']);assert record['drawEnd']>record['drawStart'];assert not record['errors']
        result[arm]=dict(long_block_median_ms_trials=timing,median_ms=float(np.median(timing)),nominal_500ms_realtime=float(500/np.median(timing)),max_coefficient_delta=max(parity),peak_physical_mib_trials=memory,allocation=allocation)
    assert all(a==result['late']['allocation'][0] for name,r in result.items() if name!='v4' for a in r['allocation'])
    assert result['early']['allocation'][0]['bytes']-result['v4']['allocation'][0]['bytes']<2**20
    checks=dict(speed_vs_late=result['early']['median_ms']<=1.10*result['late']['median_ms'],speed_vs_v4=result['early']['median_ms']<=1.25*result['v4']['median_ms'],extra_gpu_under_1mib=result['early']['allocation'][0]['bytes']-result['v4']['allocation'][0]['bytes']<2**20)
    freeze(OUT/'browser-summary.json',dict(models=result,checks={k:bool(v) for k,v in checks.items()},scope='Desktop WebKit, M5 Max64GiB, normal animated James with accelerated prerecorded STA; TTS, initial rig loading, network and browser parent excluded. Head arms share their extended graph; v4 uses its original runner. Small process-footprint differences do not prove memory savings.'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
